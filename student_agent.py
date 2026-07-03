import heapq
import re
from typing import Optional


OBJECT_CANONICAL_RE = re.compile(r"\((attack|succumb) ([a-z])\)|\((feast|overcome) ([a-z]) ([a-z])\)")
BLOCK_CANONICAL_RE = re.compile(
    r"\((engage_payload|release_payload) ([a-z]+)\)|\((unmount_node|mount_node) ([a-z]+) ([a-z]+)\)"
)


class AssemblyAgent:
    def __init__(self):
        self.system_prompt = (
            "You are a deterministic planning solver. Return only canonical actions, "
            "one action per line. No explanations."
        )

    def solve(self, scenario_context: str, llm_engine_func) -> list:
        """
        Recibe el texto del escenario y la funcion del motor LLM.
        Devuelve una lista de acciones canonicas.
        """
        domain = detect_domain(scenario_context)

        # El planificador simbolico es rapido y reproducible; Qwen queda como
        # respaldo cuando el texto no encaja con los dos dominios conocidos.
        symbolic_plan = deterministic_plan(scenario_context, domain)
        if symbolic_plan:
            return symbolic_plan

        prompt = build_prompt(scenario_context, domain)
        raw_response = llm_engine_func(
            prompt=prompt,
            system=self.system_prompt,
            max_new_tokens=192,
            temperature=0.0,
            do_sample=False,
            enable_thinking=False,
        )
        return parse_actions(raw_response, domain)


def detect_domain(scenario_context: str) -> str:
    text = scenario_context.lower()
    if "set of blocks" in text or "mount_node" in text or "pick up a block" in text:
        return "blocks"
    return "objects"


def build_prompt(scenario_context: str, domain: str) -> str:
    if domain == "blocks":
        valid_actions = """Valid canonical actions:
(engage_payload color)
(release_payload color)
(unmount_node color color)
(mount_node color color)

Natural language mapping:
pick up the red block -> (engage_payload red)
put down the red block -> (release_payload red)
unmount_node the red block from on top of the blue block -> (unmount_node red blue)
mount_node/stack the red block on top of the blue block -> (mount_node red blue)"""
    else:
        valid_actions = """Valid canonical actions:
(attack a)
(succumb a)
(feast a b)
(overcome a b)

Natural language mapping:
attack object a -> (attack a)
succumb object a -> (succumb a)
feast object a from object b -> (feast a b)
overcome object a from object b -> (overcome a b)"""

    return f"""{scenario_context}

Complete only the final empty [PLAN].
Do not repeat the solved demonstration plan.
Return only actions, one action per line.
Do not include bullets, numbering, JSON, markdown, code fences, [PLAN], [PLAN END], or explanations.
Each line must be exactly one canonical action.

{valid_actions}"""


def deterministic_plan(scenario_context: str, domain: str) -> list:
    try:
        if domain == "blocks":
            return solve_blocks(scenario_context)
        return solve_objects(scenario_context)
    except Exception:
        return []


def parse_actions(raw_text: str, domain: str) -> list:
    if not raw_text:
        return []

    text = _strip_plan_markers(raw_text)
    regex = BLOCK_CANONICAL_RE if domain == "blocks" else OBJECT_CANONICAL_RE
    actions = [_normalize_canonical(match.group(0)) for match in regex.finditer(text.lower())]

    if actions:
        return [a for a in actions if validate_action_format(a, domain)]

    parsed = []
    for line in text.splitlines():
        action = canonicalize_action(line, domain)
        if action and validate_action_format(action, domain):
            parsed.append(action)
    return parsed


def canonicalize_action(line: str, domain: str) -> Optional[str]:
    text = line.strip().lower()
    text = re.sub(r"^[-*\d\.\)\s]+", "", text)
    text = text.strip("`'\"[]{} ")
    text = text.replace("  ", " ")

    if domain == "blocks":
        patterns = [
            (r"^pick up the ([a-z]+) block$", "(engage_payload {0})"),
            (r"^put down the ([a-z]+) block$", "(release_payload {0})"),
            (
                r"^unmount_node the ([a-z]+) block from on top of the ([a-z]+) block$",
                "(unmount_node {0} {1})",
            ),
            (r"^(?:mount_node|stack) the ([a-z]+) block on top of the ([a-z]+) block$", "(mount_node {0} {1})"),
        ]
    else:
        patterns = [
            (r"^attack object ([a-z])$", "(attack {0})"),
            (r"^succumb object ([a-z])$", "(succumb {0})"),
            (r"^feast object ([a-z]) from object ([a-z])$", "(feast {0} {1})"),
            (r"^overcome object ([a-z]) from object ([a-z])$", "(overcome {0} {1})"),
        ]

    for pattern, template in patterns:
        match = re.match(pattern, text)
        if match:
            return template.format(*match.groups())
    return None


def validate_action_format(action: str, domain: str) -> bool:
    regex = BLOCK_CANONICAL_RE if domain == "blocks" else OBJECT_CANONICAL_RE
    return bool(regex.fullmatch(action))


def _normalize_canonical(action: str) -> str:
    return re.sub(r"\s+", " ", action.strip().lower())


def _strip_plan_markers(raw_text: str) -> str:
    text = raw_text.replace("```", "")
    if "[PLAN]" in text:
        text = text.split("[PLAN]", 1)[-1]
    if "[PLAN END]" in text:
        text = text.split("[PLAN END]", 1)[0]
    return text


def _last_statement(scenario_context: str) -> str:
    parts = scenario_context.split("[STATEMENT]")
    return parts[-1] if parts else scenario_context


def solve_objects(scenario_context: str) -> list:
    statement = _last_statement(scenario_context).lower()
    init_text = _between(statement, "as initial conditions i have that,", "my goal is to have that")
    goal_text = _between(statement, "my goal is to have that", "my plan is as follows")

    objects = sorted(set(re.findall(r"object ([a-z])", init_text + " " + goal_text)))
    goals = re.findall(r"object ([a-z]) craves object ([a-z])", goal_text)
    if not objects or not goals:
        return []

    init_state = set()
    if "harmony" in init_text:
        init_state.add("harmony")
    for obj in re.findall(r"planet object ([a-z])", init_text):
        init_state.add(f"planet:{obj}")
    for obj in re.findall(r"province object ([a-z])", init_text):
        init_state.add(f"province:{obj}")
    for a, b in re.findall(r"object ([a-z]) craves object ([a-z])", init_text):
        init_state.add(f"craves:{a}:{b}")

    goal_facts = frozenset(f"craves:{a}:{b}" for a, b in goals)
    if goal_facts.issubset(init_state):
        return []

    return _astar_objects(frozenset(init_state), objects, goals, goal_facts)


def _astar_objects(init_state: frozenset, objects: list, goals: list, goal_facts: frozenset) -> list:
    max_depth = 24
    queue = [(0, 0, 0, init_state, [])]
    visited = {init_state: 0}
    counter = 1

    while queue:
        _, cost, _, state, plan = heapq.heappop(queue)
        if goal_facts.issubset(state):
            return plan
        if cost >= max_depth:
            continue

        for action, next_state in _object_successors(state, objects, goals):
            new_cost = cost + 1
            if visited.get(next_state, 999) <= new_cost:
                continue
            visited[next_state] = new_cost
            next_plan = plan + [action]
            priority = new_cost + _object_heuristic(next_state, goals)
            heapq.heappush(queue, (priority, new_cost, counter, next_state, next_plan))
            counter += 1
    return []


def _object_successors(state: frozenset, objects: list, goals: list):
    successors = []
    object_order = _goal_object_order(objects, goals)
    target_order = {a: [b for x, b in goals if x == a] for a in objects}

    def add(action, new_state):
        successors.append((action, frozenset(new_state)))

    for a in object_order:
        for b in target_order.get(a, []):
            if f"pain:{a}" in state and f"province:{b}" in state:
                ns = set(state)
                ns.add("harmony")
                ns.add(f"province:{a}")
                ns.add(f"craves:{a}:{b}")
                ns.discard(f"province:{b}")
                ns.discard(f"pain:{a}")
                add(f"(overcome {a} {b})", ns)

        craves_order = target_order.get(a, []) + objects
        seen_b = set()
        for b in craves_order:
            if b in seen_b:
                continue
            seen_b.add(b)
            if f"craves:{a}:{b}" in state and f"province:{a}" in state and "harmony" in state:
                ns = set(state)
                ns.add(f"pain:{a}")
                ns.add(f"province:{b}")
                ns.discard(f"craves:{a}:{b}")
                ns.discard(f"province:{a}")
                ns.discard("harmony")
                add(f"(feast {a} {b})", ns)

        if f"pain:{a}" in state:
            ns = set(state)
            ns.add(f"province:{a}")
            ns.add(f"planet:{a}")
            ns.add("harmony")
            ns.discard(f"pain:{a}")
            add(f"(succumb {a})", ns)

        for b in objects:
            if b not in target_order.get(a, []) and f"pain:{a}" in state and f"province:{b}" in state and a != b:
                ns = set(state)
                ns.add("harmony")
                ns.add(f"province:{a}")
                ns.add(f"craves:{a}:{b}")
                ns.discard(f"province:{b}")
                ns.discard(f"pain:{a}")
                add(f"(overcome {a} {b})", ns)

        if f"province:{a}" in state and f"planet:{a}" in state and "harmony" in state:
            ns = set(state)
            ns.add(f"pain:{a}")
            ns.discard(f"province:{a}")
            ns.discard(f"planet:{a}")
            ns.discard("harmony")
            add(f"(attack {a})", ns)

    return successors


def _goal_object_order(objects: list, goals: list) -> list:
    ordered = []
    for a, _ in goals:
        if a not in ordered:
            ordered.append(a)
    for obj in objects:
        if obj not in ordered:
            ordered.append(obj)
    return ordered


def _object_heuristic(state: frozenset, goals: list) -> int:
    missing = 0
    prep = 0
    for a, b in goals:
        if f"craves:{a}:{b}" in state:
            continue
        missing += 2
        if f"pain:{a}" in state:
            missing -= 1
        elif any(f.startswith(f"craves:{a}:") for f in state) or f"planet:{a}" in state:
            prep += 1
        if f"province:{b}" not in state:
            prep += 1
    return missing + prep


def solve_blocks(scenario_context: str) -> list:
    statement = _last_statement(scenario_context).lower()
    init_text = _between(statement, "as initial conditions i have that,", "my goal is to have that")
    goal_text = _between(statement, "my goal is to have that", "my plan is as follows")

    blocks = sorted(set(re.findall(r"the ([a-z]+) block", init_text + " " + goal_text)))
    if not blocks:
        return []

    init_on = {block: "table" for block in blocks}
    for block, support in re.findall(r"the ([a-z]+) block is on top of the ([a-z]+) block", init_text):
        init_on[block] = support
    for block in re.findall(r"the ([a-z]+) block is on the table", init_text):
        init_on[block] = "table"

    goal_on = {}
    for block, support in re.findall(r"the ([a-z]+) block is on top of the ([a-z]+) block", goal_text):
        goal_on[block] = support

    if all(init_on.get(block) == support for block, support in goal_on.items()):
        return []

    return _astar_blocks(tuple(init_on[b] for b in blocks), None, blocks, goal_on)


def _astar_blocks(init_on: tuple, init_holding: Optional[str], blocks: list, goal_on: dict) -> list:
    start = (init_on, init_holding)
    max_depth = 24
    queue = [(0, 0, 0, start, [])]
    visited = {start: 0}
    counter = 1

    while queue:
        _, cost, _, state, plan = heapq.heappop(queue)
        on_tuple, holding = state
        on = dict(zip(blocks, on_tuple))
        if holding is None and all(on.get(block) == support for block, support in goal_on.items()):
            return plan
        if cost >= max_depth:
            continue

        for action, next_state in _block_successors(state, blocks, goal_on):
            new_cost = cost + 1
            if visited.get(next_state, 999) <= new_cost:
                continue
            visited[next_state] = new_cost
            next_plan = plan + [action]
            priority = new_cost + _block_heuristic(next_state, blocks, goal_on)
            heapq.heappush(queue, (priority, new_cost, counter, next_state, next_plan))
            counter += 1
    return []


def _block_successors(state, blocks: list, goal_on: dict):
    on_tuple, holding = state
    on = dict(zip(blocks, on_tuple))
    clear = _clear_blocks(on, holding, blocks)
    successors = []

    def make_state(new_on, new_holding):
        return (tuple(new_on[b] for b in blocks), new_holding)

    if holding:
        preferred_support = goal_on.get(holding)
        preferred_supports = []
        if preferred_support and preferred_support in clear:
            preferred_supports.append(preferred_support)

        for support in preferred_supports:
            new_on = dict(on)
            new_on[holding] = support
            successors.append((f"(mount_node {holding} {support})", make_state(new_on, None)))

        new_on = dict(on)
        new_on[holding] = "table"
        successors.append((f"(release_payload {holding})", make_state(new_on, None)))

        for support in _support_order(blocks, goal_on):
            if support == holding or support not in clear or support in preferred_supports:
                continue
            new_on = dict(on)
            new_on[holding] = support
            successors.append((f"(mount_node {holding} {support})", make_state(new_on, None)))
        return successors

    for block in _block_order(blocks, goal_on):
        support = on[block]
        if block not in clear:
            continue
        if support == "table":
            new_on = dict(on)
            new_on[block] = "held"
            successors.append((f"(engage_payload {block})", make_state(new_on, block)))
        else:
            new_on = dict(on)
            new_on[block] = "held"
            successors.append((f"(unmount_node {block} {support})", make_state(new_on, block)))
    return successors


def _clear_blocks(on: dict, holding: Optional[str], blocks: list) -> set:
    occupied = {support for support in on.values() if support not in ("table", "held")}
    return {block for block in blocks if block != holding and block not in occupied and on[block] != "held"}


def _block_order(blocks: list, goal_on: dict) -> list:
    ordered = []
    for block in goal_on:
        if block not in ordered:
            ordered.append(block)
    for support in goal_on.values():
        if support not in ordered:
            ordered.append(support)
    for block in blocks:
        if block not in ordered:
            ordered.append(block)
    return ordered


def _support_order(blocks: list, goal_on: dict) -> list:
    ordered = []
    for support in goal_on.values():
        if support not in ordered:
            ordered.append(support)
    for block in blocks:
        if block not in ordered:
            ordered.append(block)
    return ordered


def _block_heuristic(state, blocks: list, goal_on: dict) -> int:
    on_tuple, holding = state
    on = dict(zip(blocks, on_tuple))
    missing = sum(1 for block, support in goal_on.items() if on.get(block) != support)
    blockers = 0
    for block, support in goal_on.items():
        if on.get(block) == support:
            continue
        if block not in _clear_blocks(on, holding, blocks):
            blockers += 1
        if support not in _clear_blocks(on, holding, blocks):
            blockers += 1
    return missing + blockers + (1 if holding else 0)


def _between(text: str, start: str, end: str) -> str:
    if start in text:
        text = text.split(start, 1)[1]
    if end in text:
        text = text.split(end, 1)[0]
    return text
