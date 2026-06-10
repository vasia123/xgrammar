"""Tests for the compile_timeout_ms parameter of GrammarCompiler.

Token-mask precomputation costs FSM-states x vocab-size. Some grammars
(e.g. long chains of optional rules) blow up the state count and compile
for minutes on a real vocabulary, with no way to abort. With
compile_timeout_ms set, the compile aborts cooperatively and raises an
error of kind CompileTimeoutError instead of running to completion.

Uses a synthetic 256-byte vocabulary so the tests are self-contained
(no Hugging Face access needed).
"""

import sys
import time

import pytest

import xgrammar as xgr


def _pathological_optional_chain_grammar(n: int) -> str:
    """A grammar whose token-mask precompute explodes: a chain of n optional
    rules multiplies the number of reachable FSM states."""
    parts = ["root ::= " + " ".join(f"f{i}?" for i in range(n)) + ' "end"']
    parts.extend(f'f{i} ::= "a{i}" strchar' for i in range(n))
    parts.append("strchar ::= [a-zA-Z0-9 ]")
    return "\n".join(parts)


def _byte_vocab_tokenizer_info() -> xgr.TokenizerInfo:
    vocab = [bytes([b]) for b in range(256)]
    return xgr.TokenizerInfo(vocab, xgr.VocabType.RAW, stop_token_ids=[0])


def test_no_timeout_by_default():
    compiler = xgr.GrammarCompiler(_byte_vocab_tokenizer_info(), max_threads=2)
    compiler.compile_grammar(_pathological_optional_chain_grammar(200))


def test_generous_timeout_compiles_normally():
    compiler = xgr.GrammarCompiler(
        _byte_vocab_tokenizer_info(), max_threads=2, compile_timeout_ms=60_000
    )
    compiler.compile_grammar('root ::= "hello" [a-z]*')


def test_timeout_aborts_pathological_compile():
    compiler = xgr.GrammarCompiler(
        _byte_vocab_tokenizer_info(), max_threads=2, compile_timeout_ms=10
    )
    time_start = time.monotonic()
    with pytest.raises(Exception, match="[Cc]ompile timeout"):
        compiler.compile_grammar(_pathological_optional_chain_grammar(2000))
    # Loose bound: cooperative cancellation + CI noise, but far below the
    # multi-second full compile.
    assert time.monotonic() - time_start < 2.0


def test_timed_out_grammar_is_not_poisoned_in_cache():
    # A timeout is a property of the call, not of the grammar: the retry
    # must actually recompile (and time out again) instead of rethrowing
    # an exception cached forever, and a permissive compiler must succeed.
    grammar = _pathological_optional_chain_grammar(800)

    strict = xgr.GrammarCompiler(_byte_vocab_tokenizer_info(), max_threads=2, compile_timeout_ms=1)
    with pytest.raises(Exception, match="[Cc]ompile timeout"):
        strict.compile_grammar(grammar)
    time_start = time.monotonic()
    with pytest.raises(Exception, match="[Cc]ompile timeout"):
        strict.compile_grammar(grammar)
    # A cached-exception rethrow returns in microseconds; a real recompile
    # under a 1ms deadline takes at least the deadline.
    assert time.monotonic() - time_start >= 0.001

    permissive = xgr.GrammarCompiler(
        _byte_vocab_tokenizer_info(), max_threads=2, compile_timeout_ms=600_000
    )
    permissive.compile_grammar(grammar)


def test_valid_grammars_unaffected_after_timeout():
    compiler = xgr.GrammarCompiler(
        _byte_vocab_tokenizer_info(), max_threads=2, compile_timeout_ms=10
    )
    with pytest.raises(Exception, match="[Cc]ompile timeout"):
        compiler.compile_grammar(_pathological_optional_chain_grammar(2000))
    compiled = compiler.compile_grammar('root ::= "ok" [0-9]+')
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    assert matcher.accept_string("ok42")


if __name__ == "__main__":
    pytest.main(sys.argv)
