"""The fixed identity cases shared by verify_identity.py and make_golden.py."""
TOOL = [{"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
ORD = [{"role": "user", "content": "Explain in three sentences why the sky is blue."}]
TOOLMSG = [{"role": "user", "content": "List the files in /tmp. Use the list_dir tool."}]
SHORT = [{"role": "user", "content": "Reply with one short sentence about the moon. Do not call tools."}]
CASES = [
    ("greedy_ordinary_200", ORD, 0.0, 7, None, 200),
    ("seeded_ordinary_200", ORD, 0.7, 424242, None, 200),
    ("greedy_tools_400", TOOLMSG, 0.0, 13, TOOL, 400),
    ("seeded_tools_400", TOOLMSG, 0.7, 31337, TOOL, 400),
    ("short_greedy_ordinary", SHORT, 0.0, 7, None, 12),
    ("short_seeded_ordinary", SHORT, 0.7, 424242, None, 12),
    ("short_greedy_tools", SHORT, 0.0, 13, TOOL, 12),
    ("short_seeded_tools", SHORT, 0.7, 31337, TOOL, 12),
]
EXPECT_TOOL_CALL = {"greedy_tools_400", "seeded_tools_400"}
SEEDED_VS_GREEDY = [("seeded_ordinary_200", "greedy_ordinary_200")]
