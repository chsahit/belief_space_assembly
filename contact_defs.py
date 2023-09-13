ground_align = set(
    (
        ("bin_model::bottom", "block::111"),
        ("bin_model::bottom", "block::011"),
        ("bin_model::bottom", "block::101"),
        ("bin_model::bottom", "block::001"),
    )
)

corner_align = set(
    (
        ("bin_model::bottom", "block::111"),
        ("bin_model::bottom", "block::011"),
        ("bin_model::bottom", "block::101"),
        ("bin_model::bottom", "block::001"),
        ("bin_model::back", "block::112"),
        ("bin_model::back", "block::102"),
        ("bin_model::left", "block::102"),
    )
)

corner_align_min = set(
    (
        ("bin_model::bottom", "block::101"),
        ("bin_model::back", "block::112"),
        ("bin_model::back", "block::102"),
        ("bin_model::left", "block::102"),
    )
)

ff_align = set(
    (
        ("bin_model::bottom", "block::111"),
        ("bin_model::bottom", "block::011"),
        ("bin_model::bottom", "block::101"),
        ("bin_model::bottom", "block::001"),
        ("bin_model::back", "block::112"),
        ("bin_model::back", "block::102"),
    )
)

ff_only_align = set(
    (("bin_model::back", "block::111"), ("bin_model::back", "block::101"))
)


bf_only_align = set(
    (("bin_model::front", "block::011"), ("bin_model::front", "block::001"))
)

bc_only_align = set((("bin_model::front", "block::001"),))

lc_align = set((("bin_model::left", "block::101"),))
fc_align = set((("bin_model::back", "block::101"),))
f_align = set((("bin_model::left", "block::101"), ("bin_model::back", "block::101")))
lf_align = set((("bin_model::left", "block::101"), ("bin_model::left", "block::001")))

f_align_c = set((("bin_model::left", "block::101"), ("bin_model::right", "block::111")))

fl_chamfer_touch = set((("bin_model::back_chamfer", "block::101"),))
bl_chamfer_touch = set((("bin_model::front_chamfer", "block::011"),))
l_chamfer_touch = set((("bin_model::left_chamfer", "block::101"),))
l_full_chamfer_touch = set(
    (
        ("bin_model::left_chamfer", "block::101"),
        ("bin_model::left_chamfer", "block::001"),
    )
)
f_full_chamfer_touch = set(
    (
        ("bin_model::back_chamfer", "block::101"),
        ("bin_model::back_chamfer", "block::111"),
    )
)

corner_touch = set(
    (
        ("bin_model::back", "block::101"),
        ("bin_model::back", "block::111"),
        ("bin_model::left", "block::101"),
    )
)
top_touch = set(
    (
        ("bin_model::back", "block::101"),
        ("bin_model::back", "block::111"),
        ("bin_model::front", "block::001"),
        ("bin_model::front", "block::011"),
    )
)

corner_align_2 = set(
    (
        ("bin_model::back", "block::101"),
        ("bin_model::back", "block::111"),
        ("bin_model::back", "block::102"),
        ("bin_model::left", "block::001"),
        ("bin_model::left", "block::101"),
        ("bin_model::left", "block::102"),
    )
)

corner_align_3 = set(
    (
        ("bin_model::back", "block::101"),
        ("bin_model::left", "block::101"),
        ("bin_model::back", "block::102"),
        ("bin_model::left", "block::102"),
    )
)
