top_touch = set((("big_fixed_puzzle::b3_top", "block::b4_bottom"),))
bt = set((("big_fixed_puzzle::b4_bottom", "block::b5_top"),))
# bottom = set((("big_fixed_puzzle::b1_top", "block::b3_bottom"),))
bottom = set(
    (
        ("big_fixed_puzzle::b1_top", "block::b3_bottom"),
        ("big_fixed_puzzle::b1_front", "block::b4_back"),
    )
)
side = set((("big_fixed_puzzle::b2_inside", "block::b2"),))
goal = set(
    (
        ("big_fixed_puzzle::b1_top", "block::b3"),
        ("big_fixed_puzzle::b2_inside", "block::b2"),
    )
)
