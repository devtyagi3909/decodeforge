"""Deterministic Finite Automaton (DFA) engine for regex-guided decoding."""

from __future__ import annotations


class RegexDFA:
    """Deterministic Finite Automaton constructed from a regular expression."""

    def __init__(
        self,
        states: set[int],
        transitions: dict[tuple[int, str], int],
        start_state: int,
        accept_states: set[int],
    ) -> None:
        self.states = states
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
        self._dead_state: int | None = None

    def is_accepting(self, state: int) -> bool:
        """Return True if the given state is an accepting (terminal) state."""
        return state in self.accept_states

    def transition(self, state: int, char: str) -> int | None:
        """Advance one character from state. Returns next state or None if invalid."""
        return self.transitions.get((state, char))

    def walk_string(self, start_state: int, text: str) -> int | None:
        """Advance across a sequence of characters.

        Returns the reached state, or None if any character transition fails.
        """
        curr = start_state
        for char in text:
            nxt = self.transitions.get((curr, char))
            if nxt is None:
                return None
            curr = nxt
        return curr

    @classmethod
    def from_pattern(cls, pattern: str) -> RegexDFA:
        """Compile a regular expression pattern into a RegexDFA."""
        builder = _DFACompiler(pattern)
        return builder.compile()


# -----------------------------------------------------------------------------
# Regex AST & Compiler (Thompson NFA -> Subset Construction DFA)
# -----------------------------------------------------------------------------


class _NFAState:
    __slots__ = ("epsilon_transitions", "transitions")

    def __init__(self) -> None:
        self.transitions: dict[str, set[_NFAState]] = {}
        self.epsilon_transitions: set[_NFAState] = set()

    def add_char(self, char: str, target: _NFAState) -> None:
        if char not in self.transitions:
            self.transitions[char] = set()
        self.transitions[char].add(target)

    def add_epsilon(self, target: _NFAState) -> None:
        self.epsilon_transitions.add(target)


class _NFAGraph:
    def __init__(self, start: _NFAState, end: _NFAState) -> None:
        self.start = start
        self.end = end


class _DFACompiler:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0

    def compile(self) -> RegexDFA:
        # 1. Parse pattern into NFA
        nfa = self._parse_regex()
        # 2. Convert NFA to DFA via subset construction
        return self._nfa_to_dfa(nfa)

    def _peek(self) -> str | None:
        if self.pos < len(self.pattern):
            return self.pattern[self.pos]
        return None

    def _next(self) -> str:
        ch = self.pattern[self.pos]
        self.pos += 1
        return ch

    def _parse_regex(self) -> _NFAGraph:
        # alternation: term ('|' term)*
        terms = [self._parse_term()]
        while self._peek() == "|":
            self._next()
            terms.append(self._parse_term())

        if len(terms) == 1:
            return terms[0]

        start = _NFAState()
        end = _NFAState()
        for t in terms:
            start.add_epsilon(t.start)
            t.end.add_epsilon(end)
        return _NFAGraph(start, end)

    def _parse_term(self) -> _NFAGraph:
        # concatenation of factors
        factors = []
        while self.pos < len(self.pattern) and self._peek() not in ("|", ")"):
            factors.append(self._parse_factor())

        if not factors:
            # Empty match epsilon
            s = _NFAState()
            e = _NFAState()
            s.add_epsilon(e)
            return _NFAGraph(s, e)

        for i in range(len(factors) - 1):
            factors[i].end.add_epsilon(factors[i + 1].start)
        return _NFAGraph(factors[0].start, factors[-1].end)

    def _parse_factor(self) -> _NFAGraph:
        base = self._parse_atom()
        # Quantifiers: *, +, ?
        while self._peek() in ("*", "+", "?"):
            q = self._next()
            if q == "*":
                s = _NFAState()
                e = _NFAState()
                s.add_epsilon(base.start)
                s.add_epsilon(e)
                base.end.add_epsilon(base.start)
                base.end.add_epsilon(e)
                base = _NFAGraph(s, e)
            elif q == "+":
                s = _NFAState()
                e = _NFAState()
                s.add_epsilon(base.start)
                base.end.add_epsilon(base.start)
                base.end.add_epsilon(e)
                base = _NFAGraph(s, e)
            elif q == "?":
                s = _NFAState()
                e = _NFAState()
                s.add_epsilon(base.start)
                s.add_epsilon(e)
                base.end.add_epsilon(e)
                base = _NFAGraph(s, e)
        return base

    def _parse_atom(self) -> _NFAGraph:
        ch = self._peek()
        if ch == "(":
            self._next()
            # Handle non-capturing (?:...)
            if self._peek() == "?":
                self._next()
                if self._peek() == ":":
                    self._next()
            sub = self._parse_regex()
            if self._peek() == ")":
                self._next()
            return sub
        elif ch == "[":
            return self._parse_char_class()
        elif ch == "\\":
            self._next()  # Consume \
            esc = self._next()
            return self._make_escaped_nfa(esc)
        elif ch == ".":
            self._next()
            return self._make_wildcard_nfa()
        else:
            self._next()
            s = _NFAState()
            e = _NFAState()
            s.add_char(ch, e)
            return _NFAGraph(s, e)

    def _parse_char_class(self) -> _NFAGraph:
        self._next()  # Consume [
        negated = False
        if self._peek() == "^":
            negated = True
            self._next()

        chars: set[str] = set()
        while self.pos < len(self.pattern) and self._peek() != "]":
            c = self._next()
            if c == "\\" and self.pos < len(self.pattern):
                esc = self._next()
                chars.update(self._expand_escaped_set(esc))
            elif (
                self._peek() == "-"
                and self.pos + 1 < len(self.pattern)
                and self.pattern[self.pos + 1] != "]"
            ):
                self._next()  # Consume -
                end_c = self._next()
                for codepoint in range(ord(c), ord(end_c) + 1):
                    chars.add(chr(codepoint))
            else:
                chars.add(c)

        if self._peek() == "]":
            self._next()

        # Printable ASCII universe (and common JSON characters)
        universe = [chr(i) for i in range(32, 127)] + ["\n", "\t", "\r", " "]
        if negated:
            allowed = set(universe) - chars
        else:
            allowed = chars

        s = _NFAState()
        e = _NFAState()
        for char in allowed:
            s.add_char(char, e)
        return _NFAGraph(s, e)

    def _make_wildcard_nfa(self) -> _NFAGraph:
        s = _NFAState()
        e = _NFAState()
        # Any printable ASCII + whitespace
        for i in range(32, 127):
            s.add_char(chr(i), e)
        for ws in (" ", "\t", "\n"):
            s.add_char(ws, e)
        return _NFAGraph(s, e)

    def _expand_escaped_set(self, esc: str) -> set[str]:
        if esc == "d":
            return set("0123456789")
        elif esc == "s":
            return set(" \t\n\r")
        elif esc == "w":
            res = set("0123456789_")
            for i in range(ord("a"), ord("z") + 1):
                res.add(chr(i))
            for i in range(ord("A"), ord("Z") + 1):
                res.add(chr(i))
            return res
        elif esc == "n":
            return {"\n"}
        elif esc == "t":
            return {"\t"}
        elif esc == "r":
            return {"\r"}
        return {esc}

    def _make_escaped_nfa(self, esc: str) -> _NFAGraph:
        chars = self._expand_escaped_set(esc)
        s = _NFAState()
        e = _NFAState()
        for char in chars:
            s.add_char(char, e)
        return _NFAGraph(s, e)

    def _epsilon_closure(self, states: set[_NFAState]) -> frozenset[_NFAState]:
        closure = set(states)
        stack = list(states)
        while stack:
            st = stack.pop()
            for nxt in st.epsilon_transitions:
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)
        return frozenset(closure)

    def _nfa_to_dfa(self, nfa: _NFAGraph) -> RegexDFA:
        start_closure = self._epsilon_closure({nfa.start})
        dfa_state_map: dict[frozenset[_NFAState], int] = {start_closure: 0}
        states: set[int] = {0}
        transitions: dict[tuple[int, str], int] = {}
        accept_states: set[int] = set()

        if nfa.end in start_closure:
            accept_states.add(0)

        unmarked = [start_closure]

        while unmarked:
            curr_set = unmarked.pop(0)
            curr_id = dfa_state_map[curr_set]

            # Find all possible character transitions
            char_moves: dict[str, set[_NFAState]] = {}
            for nfa_state in curr_set:
                for char, nxt_states in nfa_state.transitions.items():
                    if char not in char_moves:
                        char_moves[char] = set()
                    char_moves[char].update(nxt_states)

            for char, target_nfa_states in char_moves.items():
                target_closure = self._epsilon_closure(target_nfa_states)
                if not target_closure:
                    continue

                if target_closure not in dfa_state_map:
                    new_id = len(dfa_state_map)
                    dfa_state_map[target_closure] = new_id
                    states.add(new_id)
                    unmarked.append(target_closure)
                    if nfa.end in target_closure:
                        accept_states.add(new_id)

                target_id = dfa_state_map[target_closure]
                transitions[(curr_id, char)] = target_id

        return RegexDFA(states, transitions, 0, accept_states)
