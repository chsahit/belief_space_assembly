top_touch = set((("big_fixed_puzzle::b3_top", "block::b4_bottom"),))
top_touch2 = set(
    (
        ("big_fixed_puzzle::b3_back", "block::b4_bottom"),
        ("big_fixed_puzzle::b3_back", "block::b4_left"),
        ("big_fixed_puzzle::b3_back", "block::b4_right"),
        ("big_fixed_puzzle::b3_top", "block::b4_bottom"),
    )
)
# bt = set((("big_fixed_puzzle::b4_bottom", "block::b5_top"),))
# bt = set((("big_fixed_puzzle::b4_front", "block::b5_back"),))
bt = set(
    (
        ("big_fixed_puzzle::b4_front", "block::b5_back"),
        ("big_fixed_puzzle::b2_left", "block::b5_right"),
    )
)

bt4 = set(
    (
        ("big_fixed_puzzle::b4_front", "block::b5_top"),
        ("big_fixed_puzzle::b5_right", "block::b4_left"),
    )
)

bt4_check = set((("big_fixed_puzzle::b4", "block::301"),))
mid_depth = set(
    (
        ("big_fixed_puzzle::b4_front", "block::b3_back"),
        ("big_fixed_puzzle::b2_left", "block::b3_right"),
    )
)

bt3 = set((("big_fixed_puzzle::b4_front", "block::b3_back"),))
bt2 = set((("big_fixed_puzzle::b3_back", "block::b4_front"),))
# bottom = set((("big_fixed_puzzle::b1_top", "block::b3_bottom"),))
bottom = set(
    (
        ("big_fixed_puzzle::b1_top", "block::b3_bottom"),
        ("big_fixed_puzzle::b1_front", "block::b4_back"),
    )
)
s_bottom = set((("big_fixed_puzzle::b1_top", "block::b3_bottom"),))

# side = set((("big_fixed_puzzle::b2_inside", "block::b2"),))
goal = set(
    (
        ("big_fixed_puzzle::b1_top", "block::b3_bottom"),
        ("big_fixed_puzzle::b1_front", "block::b4_back"),
        ("big_fixed_puzzle::b2_left", "block::b2_right"),
    )
)

relaxations = {frozenset(bt4): bt4_check}
