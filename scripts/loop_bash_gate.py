#!/usr/bin/env python3
"""PreToolUse(Bash) hook for the research-loop session (research/agent_settings.json).

Auto-approves compound shell commands built only from a small whitelist of
read-only / bookkeeping commands, with every path confined to the repo or
../wm-worktrees. Everything else gets no decision, so the normal permission
rules (allow/deny/ask) apply unchanged. It never returns "deny" and never
widens what the deny list forbids: anything resembling a denied command
falls through to the normal flow.

The main reason it exists: Claude Code prompts for every `cd <dir> && git ...`,
and the loop works in git worktrees outside the project directory.
"""
import json
import os
import re
import shlex
import sys

REPO = "/home/appeon_a_lew/adl4cv-worlmodels"
WT = "/home/appeon_a_lew/wm-worktrees"
ROOTS = (REPO, WT)

SAFE_SIMPLE = {"ls", "cat", "head", "tail", "grep", "wc", "uniq", "diff", "jq", "cut", "tr",
               "echo", "pwd", "date", "basename", "dirname", "realpath", "test", "true", "sort"}
GIT_SAFE = {"status", "diff", "log", "show", "add", "commit", "rev-parse", "ls-files", "grep"}
SAFE_REDIRECT = re.compile(r"\s(?:[12]?>\s*/dev/null|2>&1)(?=\s|$)")
SEP = {"&&", "||", ";", "|"}


def within(path, cwd, roots=ROOTS):
    p = os.path.realpath(os.path.join(cwd, os.path.expanduser(path)))
    return any(p == r or p.startswith(r + "/") for r in roots), p


def unquoted_newlines_to_semicolons(cmd):
    out, q, esc = [], None, False
    for ch in cmd:
        if esc:
            esc = False
        elif ch == "\\" and q != "'":
            esc = True
        elif q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "\n":
            out.append(" ; ")
            continue
        out.append(ch)
    return None if q else "".join(out)


def segments(cmd):
    cmd = unquoted_newlines_to_semicolons(cmd)
    if cmd is None:
        return None
    cmd = SAFE_REDIRECT.sub(" ", " " + cmd)
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=";&|<>()")
    lex.whitespace_split = True
    try:
        toks = list(lex)
    except ValueError:
        return None
    segs, cur = [], []
    for t in toks:
        if t in SEP:
            segs.append(cur)
            cur = []
        elif set(t) <= set(";&|<>()"):
            return None  # redirection, subshell, background job, ...
        else:
            cur.append(t)
    segs.append(cur)
    return [s for s in segs if s]


def git_ok(args, cwd):
    while args and args[0] == "-C":
        if len(args) < 2 or not within(args[1], cwd)[0]:
            return False
        cwd = within(args[1], cwd)[1]
        args = args[2:]
    if not args or args[0].startswith("-"):
        return False  # -c key=val, --git-dir, --exec-path, ...
    sub, rest = args[0], args[1:]
    if sub in GIT_SAFE:
        return not any(a.startswith("--output") or a in ("-O", "--open-files-in-pager") or a.startswith("-O")
                       for a in rest)
    if sub == "worktree":
        return bool(rest) and rest[0] in ("add", "list")
    if sub == "branch":
        return not any(a.startswith("-") and a not in ("-a", "--list", "-v", "-vv", "--show-current") for a in rest)
    return False


def seg_ok(seg, cwd):
    """Return (ok, new_cwd)."""
    head, args = seg[0], seg[1:]
    if head == "cd":
        if len(args) != 1:
            return False, cwd
        ok, p = within(args[0], cwd)
        return ok, p
    if head == "git":
        return git_ok(args, cwd), cwd
    if re.search(r"(^|/)scripts/agent\.sh$", head):
        p = os.path.realpath(os.path.join(cwd, head))
        return p == f"{REPO}/scripts/agent.sh" or re.fullmatch(rf"{WT}/[^/]+/scripts/agent\.sh", p) is not None, cwd
    if head == "sed":
        return (len(args) >= 2 and args[0] == "-n" and re.fullmatch(r"[0-9]*(,[0-9$]*)?p", args[1]) is not None
                and all(within(a, cwd)[0] for a in args[2:])), cwd
    if head == "sort":
        return not any(a.startswith("-o") or a.startswith("--output") for a in args), cwd
    if head == "mkdir":
        return bool(args) and args[0] == "-p" and all(within(a, cwd)[0] for a in args[1:]), cwd
    if head == "cp":
        paths = [a for a in args if not a.startswith("-")]
        if len(paths) < 2 or any(a not in ("-r", "-p") for a in args if a.startswith("-")):
            return False, cwd
        dest_ok, dest = within(paths[-1], cwd)
        research = dest.startswith(f"{REPO}/research/") or re.match(rf"{WT}/[^/]+/research/", dest + "/")
        return dest_ok and bool(research) and all(within(a, cwd)[0] for a in paths[:-1]), cwd
    if head in SAFE_SIMPLE:
        return True, cwd
    return False, cwd


def decide(cmd, cwd):
    if "$(" in cmd or "`" in cmd or "<<" in cmd:
        return False
    segs = segments(cmd)
    if not segs:
        return False
    for seg in segs:
        ok, cwd = seg_ok(seg, cwd)
        if not ok:
            return False
    return True


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if data.get("tool_name") != "Bash":
        return
    cmd = data.get("tool_input", {}).get("command", "")
    cwd = data.get("cwd") or REPO
    if decide(cmd, cwd):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": "loop_bash_gate: whitelisted commands within repo/worktrees"}}))


if __name__ == "__main__":
    main()
