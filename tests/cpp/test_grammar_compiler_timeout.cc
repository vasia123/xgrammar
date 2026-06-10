/*!
 *  Copyright (c) 2026 by Contributors
 * \file tests/cpp/test_grammar_compiler_timeout.cc
 * \brief Tests for the GrammarCompiler compile timeout.
 *
 * Token-mask precompilation costs FSM-states x vocab-size. Some
 * innocent-looking grammars (e.g. a chain of optional rules) blow up the
 * state count and compile for minutes on a real vocabulary, with no way
 * for the caller to abort. The `compile_timeout_ms` constructor parameter
 * bounds every Compile* call; on expiry the compile aborts cooperatively
 * and throws CompileTimeoutError instead of running to completion.
 */

#include <gtest/gtest.h>
#include <xgrammar/xgrammar.h>

#include <chrono>
#include <string>
#include <vector>

using namespace xgrammar;

namespace {

/*!
 * \brief A grammar whose token-mask precompute explodes: a chain of n
 * optional rules multiplies the number of reachable FSM states, and every
 * state is matched against the whole vocabulary.
 */
std::string PathologicalOptionalChainGrammar(int n) {
  std::string ebnf = "root ::= ";
  for (int i = 0; i < n; ++i) {
    ebnf += "f" + std::to_string(i) + "? ";
  }
  ebnf += "\"end\"\n";
  for (int i = 0; i < n; ++i) {
    ebnf += "f" + std::to_string(i) + " ::= \"a" + std::to_string(i) + "\" strchar\n";
  }
  ebnf += "strchar ::= [a-zA-Z0-9 ]\n";
  return ebnf;
}

/*! \brief 256 single-byte tokens — raw byte-level vocabulary. */
TokenizerInfo ByteVocabTokenizerInfo() {
  std::vector<std::string> vocab;
  vocab.reserve(256);
  for (int b = 0; b < 256; ++b) {
    vocab.push_back(std::string(1, static_cast<char>(b)));
  }
  return TokenizerInfo(
      vocab, VocabType::RAW, /*vocab_size=*/std::nullopt, /*stop_token_ids=*/{{0}}
  );
}

double MillisSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

TEST(GrammarCompilerTimeoutTest, NoTimeoutByDefault) {
  // Default constructor arguments: no deadline, pathological grammar
  // compiles to completion (n is kept small enough to stay test-friendly).
  auto compiler = GrammarCompiler(ByteVocabTokenizerInfo(), /*max_threads=*/2);
  auto grammar = compiler.CompileGrammar(PathologicalOptionalChainGrammar(200));
  (void)grammar;
}

TEST(GrammarCompilerTimeoutTest, GenerousTimeoutCompilesNormally) {
  auto compiler = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/2,
      /*cache_enabled=*/true,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/60'000
  );
  auto grammar = compiler.CompileGrammar("root ::= \"hello\" [a-z]*");
  (void)grammar;
}

TEST(GrammarCompilerTimeoutTest, TimeoutAbortsPathologicalCompile) {
  // n=2000 takes multiple seconds without a deadline; the 10ms deadline
  // must abort it far before completion. The latency bound is loose
  // (cooperative cancellation + CI noise) but far below full compile time.
  auto compiler = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/2,
      /*cache_enabled=*/true,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/10
  );
  auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(
      compiler.CompileGrammar(PathologicalOptionalChainGrammar(2000)), CompileTimeoutError
  );
  EXPECT_LT(MillisSince(start), 2000.0);
}

TEST(GrammarCompilerTimeoutTest, TimeoutAbortsSingleThreadedCompile) {
  // max_threads=1 takes the no-thread-pool path; the deadline must work
  // there too.
  auto compiler = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/1,
      /*cache_enabled=*/true,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/10
  );
  auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(
      compiler.CompileGrammar(PathologicalOptionalChainGrammar(2000)), CompileTimeoutError
  );
  EXPECT_LT(MillisSince(start), 2000.0);
}

TEST(GrammarCompilerTimeoutTest, TimeoutAbortsWithCacheDisabled) {
  // With the cache disabled the token-mask phase is a smaller fraction of
  // the total compile time on this tiny vocabulary, so use a larger n to
  // guarantee the mask phase starts well after the deadline has expired.
  auto compiler = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/2,
      /*cache_enabled=*/false,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/10
  );
  EXPECT_THROW(
      compiler.CompileGrammar(PathologicalOptionalChainGrammar(5000)), CompileTimeoutError
  );
}

TEST(GrammarCompilerTimeoutTest, TimedOutKeyIsNotPoisonedInCache) {
  // A timeout is a property of the call, not of the grammar: after a
  // timed-out compile, the same key must be recompilable (the cached
  // exception must not persist). Compile a small grammar under an
  // impossibly short deadline first, then verify it succeeds with the
  // deadline gone.
  //
  // To make the first call reliably exceed the deadline even on a fast
  // machine, use a grammar whose mask work starts only after the 1ms
  // deadline has long expired (~30ms total at n=800 on a dev machine).
  auto grammar_src = PathologicalOptionalChainGrammar(800);

  auto strict = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/2,
      /*cache_enabled=*/true,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/1
  );
  EXPECT_THROW(strict.CompileGrammar(grammar_src), CompileTimeoutError);

  // Same compiler instance, same key: the retry must actually recompile
  // (and time out again) rather than rethrow instantly from the cache.
  // A cached-exception rethrow returns in microseconds; a real recompile
  // under a 1ms deadline takes at least the deadline.
  auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(strict.CompileGrammar(grammar_src), CompileTimeoutError);
  EXPECT_GE(MillisSince(start), 1.0);

  // A permissive compiler over the same vocabulary compiles the same
  // grammar fine — nothing about the grammar itself is poisoned.
  auto permissive = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/2,
      /*cache_enabled=*/true,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/600'000
  );
  auto grammar = permissive.CompileGrammar(grammar_src);
  (void)grammar;
}

TEST(GrammarCompilerTimeoutTest, ValidGrammarsUnaffectedAfterTimeout) {
  // One timed-out compile must not degrade subsequent unrelated compiles
  // on the same compiler.
  auto compiler = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/2,
      /*cache_enabled=*/true,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/10
  );
  EXPECT_THROW(
      compiler.CompileGrammar(PathologicalOptionalChainGrammar(2000)), CompileTimeoutError
  );
  auto grammar = compiler.CompileGrammar("root ::= \"ok\" [0-9]+");
  (void)grammar;
}

TEST(GrammarCompilerTimeoutTest, CompileTimeoutErrorType) {
  auto compiler = GrammarCompiler(
      ByteVocabTokenizerInfo(),
      /*max_threads=*/2,
      /*cache_enabled=*/true,
      /*max_memory_bytes=*/-1,
      /*compile_timeout_ms=*/10
  );
  try {
    compiler.CompileGrammar(PathologicalOptionalChainGrammar(2000));
    FAIL() << "expected CompileTimeoutError";
  } catch (const CompileTimeoutError& e) {
    EXPECT_EQ(e.GetType(), "CompileTimeoutError");
    // The message should mention the configured timeout to be actionable.
    EXPECT_NE(std::string(e.what()).find("10"), std::string::npos) << e.what();
  }
}
