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
