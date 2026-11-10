"""Planning contracts; no torch, ComfyUI, video decoder, or GPU is required."""

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "planner.py"
SPEC = importlib.util.spec_from_file_location("h3_window_sampler_planner_test", MODULE_PATH)
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)
make_plan = planner.make_plan


class PlannerTests(unittest.TestCase):
    def test_existing_three_window_schedule(self):
        plan = make_plan(328)
        self.assertEqual([(w.start, w.frames, w.overlap) for w in plan.windows],
                         [(0, 124, 0), (102, 124, 22), (204, 124, 22)])
        self.assertEqual((plan.output_frames, plan.sampled_frames, plan.pad_frames), (328, 328, 0))

    def test_threshold_uses_new_frames_not_source_slice_with_overlap(self):
        plan = make_plan(154)
        self.assertEqual(plan.window_count, 1)
        self.assertEqual((plan.output_frames, plan.dropped_frames, plan.tail_action),
                         (124, 30, "dropped"))

    def test_exact_threshold_keeps_minimum_ninety_frame_tail(self):
        plan = make_plan(164)
        tail = plan.windows[-1]
        self.assertEqual((tail.start, tail.frames, tail.real_frames, tail.pad_frames, tail.overlap),
                         (102, 90, 62, 28, 22))
        self.assertEqual((plan.output_frames, plan.sampled_frames, plan.pad_frames), (164, 192, 28))

    def test_tail_is_large_enough_when_overlap_exceeds_ninety(self):
        plan = make_plan(300, window_frames=260, minimum_overlap=100)
        tail = plan.windows[-1]
        self.assertEqual((tail.start, tail.frames, tail.real_frames, tail.overlap, tail.pad_frames),
                         (153, 158, 147, 107, 11))
        self.assertGreater(tail.end, plan.windows[0].end)

    def test_short_source_kept_or_clearly_rejected(self):
        plan = make_plan(40)
        self.assertEqual((plan.windows[0].frames, plan.pad_frames, plan.output_frames), (90, 50, 40))
        with self.assertRaisesRegex(ValueError, "No windows remain"):
            make_plan(30)

    def test_threshold_zero_does_not_append_a_window_without_new_source(self):
        plan = make_plan(124, tail_keep_threshold=0)
        self.assertEqual(plan.window_count, 1)
        plan = make_plan(125, tail_keep_threshold=0)
        self.assertEqual(plan.window_count, 2)
        self.assertEqual(plan.windows[-1].real_frames, 23)

    def test_source_equal_to_first_window_is_not_dropped_by_large_threshold(self):
        plan = make_plan(90, window_frames=90, tail_keep_threshold=100)
        self.assertEqual(plan.output_frames, 90)

    def test_large_threshold_does_not_discard_a_complete_next_target_window(self):
        for mode, lengths in (("uniform", ""), ("manual", [90, 90])):
            with self.subTest(mode=mode):
                plan = make_plan(150, mode=mode, window_frames=90, manual_lengths=lengths,
                                 tail_keep_threshold=100)
                self.assertEqual(plan.window_count, 2)
                self.assertEqual((plan.output_frames, plan.dropped_frames), (141, 9))

    def test_discarded_tail_does_not_require_a_valid_unused_boundary(self):
        plan = make_plan(120, window_frames=90, minimum_overlap=40)
        self.assertEqual((plan.window_count, plan.dropped_frames), (1, 30))

    def test_all_supported_length_residues_align_to_start_grid(self):
        for length, overlap in ((90, 39), (107, 56), (124, 22), (141, 39), (158, 56), (175, 22)):
            with self.subTest(length=length):
                total = 2 * length - overlap
                plan = make_plan(total, window_frames=length)
                self.assertEqual(plan.window_count, 2)
                self.assertEqual(plan.windows[1].overlap, overlap)
                self.assertEqual(plan.windows[1].start % 51, 0)

    def test_there_is_no_old_seven_second_or_158_frame_limit(self):
        for frames in (175, 345, 2045, 90 + 17 * 2400):
            with self.subTest(frames=frames):
                plan = make_plan(frames, window_frames=frames)
                self.assertEqual(plan.windows[0].frames, frames)

    def test_minimum_overlap_too_large_reports_the_boundary(self):
        with self.assertRaisesRegex(ValueError, "minimum_overlap=40 is too large"):
            make_plan(200, window_frames=90, minimum_overlap=40)

    def test_manual_variable_windows(self):
        plan = make_plan(396, mode="manual", manual_lengths="124, 175, 141")
        self.assertEqual([(w.start, w.frames, w.overlap) for w in plan.windows],
                         [(0, 124, 0), (102, 175, 22), (255, 141, 22)])

    def test_manual_short_following_window_cannot_fit_required_overlap(self):
        with self.assertRaisesRegex(ValueError, "requested length 90 cannot advance"):
            make_plan(400, mode="manual", manual_lengths=[260, 90], minimum_overlap=100)

    def test_manual_lengths_are_not_repeated_implicitly(self):
        with self.assertRaisesRegex(ValueError, "exhausted.*40 uncovered"):
            make_plan(164, mode="manual", manual_lengths=[124])
        plan = make_plan(164, mode="manual", manual_lengths=[124, 124])
        self.assertEqual(plan.windows[-1].frames, 90)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            make_plan(1000, mode="manual", manual_lengths=[124, 124])

    def test_manual_drop_allows_no_extra_item_or_one_final_item(self):
        for lengths in ([124], [124, 124]):
            with self.subTest(lengths=lengths):
                plan = make_plan(154, mode="manual", manual_lengths=lengths)
                self.assertEqual((plan.window_count, plan.dropped_frames), (1, 30))
        with self.assertRaisesRegex(ValueError, "unused"):
            make_plan(154, mode="manual", manual_lengths=[124, 124, 124])

    def test_manual_unused_items_are_not_allowed_after_exact_end_or_kept_tail(self):
        for total, lengths in ((124, [124, 124]), (164, [124, 124, 124])):
            with self.subTest(total=total):
                with self.assertRaisesRegex(ValueError, "unused"):
                    make_plan(total, mode="manual", manual_lengths=lengths)

    def test_manual_parser_and_invalid_input(self):
        self.assertEqual(planner.parse_manual_lengths("90, 107\n124；141，158 175"),
                         (90, 107, 124, 141, 158, 175))
        for bad in ("90, 124.0", "90, invalid", [90, True], [90, 123], [73], [90.0]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    planner.parse_manual_lengths(bad)
        for kwargs in ({"source_frames": 0}, {"source_frames": 10.0},
                       {"source_frames": 100, "window_frames": 100},
                       {"source_frames": 100, "minimum_overlap": -1},
                       {"source_frames": 100, "tail_keep_threshold": -1},
                       {"source_frames": 100, "mode": "unknown"},
                       {"source_frames": 100, "mode": "manual", "manual_lengths": ""}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    make_plan(**kwargs)

    def test_many_source_lengths_preserve_timeline_and_crop_contracts(self):
        for length in (90, 124, 175, 260):
            for total in range(40, 701, 7):
                with self.subTest(length=length, total=total):
                    plan = make_plan(total, window_frames=length)
                    previous = None
                    for window in plan.windows:
                        self.assertEqual(window.start % 51, 0)
                        self.assertGreaterEqual(window.frames, 90)
                        self.assertEqual(window.frames % 17, 5)
                        self.assertEqual(window.frames, window.real_frames + window.pad_frames)
                        self.assertLessEqual(window.frames, window.requested_frames)
                        if previous is not None:
                            self.assertEqual(window.overlap, previous.end - window.start)
                            self.assertGreaterEqual(window.overlap, 22)
                            self.assertGreater(window.end, previous.end)
                            self.assertGreater(window.start, previous.start)
                        previous = window
                    self.assertEqual(plan.output_frames + plan.dropped_frames, total)
                    self.assertEqual(plan.sampled_frames - plan.pad_frames, plan.output_frames)
                    self.assertLess(plan.dropped_frames, 40)
                    self.assertTrue(all(w.pad_frames == 0 for w in plan.windows[:-1]))

    def test_report_is_json_ready(self):
        import json
        report = make_plan(164).as_dict()
        self.assertEqual(report["window_count"], 2)
        self.assertEqual(report["fps"], 24)
        self.assertEqual(report["windows"][-1]["pad_frames"], 28)
        json.dumps(report)

    def test_target_longer_than_source_plans_actual_frames_only(self):
        # 1000 requested, 260 real, 124 per window, tail < 40 dropped -> 2 windows.
        plan = make_plan(260, target_frames=1000)
        self.assertEqual(plan.window_count, 2)
        self.assertEqual([(w.start, w.frames) for w in plan.windows], [(0, 124), (102, 124)])
        reference = make_plan(260)
        self.assertEqual(plan.windows, reference.windows)
        self.assertEqual((plan.output_frames, plan.dropped_frames), (226, 34))
        self.assertEqual((plan.source_frames, plan.target_frames, plan.shortfall_frames), (260, 1000, 740))
        self.assertEqual(plan.as_dict()["shortfall_frames"], 740)

    def test_target_not_longer_than_source_has_no_shortfall(self):
        for target in (None, 100, 260):
            plan = make_plan(260, target_frames=target)
            self.assertEqual(plan.shortfall_frames, 0)
        self.assertEqual(make_plan(328, target_frames=124).window_count, 1)


class MixedOverlapTests(unittest.TestCase):
    def test_reachable_overlaps_are_a_discrete_set(self):
        self.assertEqual(planner.reachable_overlaps(124), (22, 73))
        self.assertEqual(planner.reachable_overlaps(175), (22, 73, 124))
        self.assertEqual(planner.reachable_overlaps(90), (39,))

    def test_explicit_overlaps_drive_each_boundary(self):
        plan = make_plan(328, overlaps=(73, 22, 73))
        self.assertEqual([(w.start, w.frames) for w in plan.windows],
                         [(0, 124), (51, 124), (153, 124), (204, 124)])
        self.assertEqual(plan.overlaps, (0, 73, 22, 73))
        self.assertEqual(plan.as_dict()["overlaps"], [0, 73, 22, 73])
        self.assertEqual(plan.pad_frames, 0)

    def test_omitting_overlaps_matches_the_previous_scalar_behaviour(self):
        for total in (124, 154, 260, 328, 396, 700):
            for overlap in (0, 22, 51, 73):
                for frames in (90, 124, 175):
                    try:
                        scalar = make_plan(total, window_frames=frames, minimum_overlap=overlap)
                    except ValueError:
                        continue
                    explicit = make_plan(total, window_frames=frames,
                                         overlaps=tuple(w.overlap for w in scalar.windows[1:]))
                    self.assertEqual(scalar.windows, explicit.windows, (total, overlap, frames))

    def test_unreachable_overlap_is_rejected_instead_of_snapped(self):
        with self.assertRaisesRegex(ValueError, "asks for an overlap of 40 frames"):
            make_plan(328, overlaps=(40, 22, 73))

    def test_overlap_list_length_must_match_the_boundary_count(self):
        with self.assertRaisesRegex(ValueError, "implies 3 window"):
            make_plan(328, overlaps=(73, 22))
        with self.assertRaisesRegex(ValueError, "implies 6 window"):
            make_plan(328, overlaps=(73, 22, 73, 73, 73))

    def test_three_prompts_cannot_be_cheapened_on_328_frames(self):
        # Six windows are unreachable and five only fit with every boundary at 73.
        self.assertIsNone(planner.solve_overlaps(328, 124, 6, [False] * 5))
        solution = planner.solve_overlaps(328, 124, 5, [False] * 4)
        self.assertEqual(solution.gaps, (73, 73, 73, 73))
        self.assertEqual(solution.matches, 4)

    def test_four_prompts_get_one_cheap_identical_prompt_boundary(self):
        solution = planner.solve_overlaps(328, 124, 4, [False, True, False])
        self.assertEqual(solution.gaps, (73, 22, 73))
        self.assertEqual((solution.matches, solution.pad_frames), (3, 0))
        self.assertEqual(solution.window_frames_total, 496)
        plan = make_plan(328, overlaps=solution.gaps)
        self.assertEqual(plan.window_count, 4)
        self.assertEqual(plan.overlaps, (0, 73, 22, 73))

    def test_three_prompts_cannot_honour_the_preference(self):
        # Exactly three windows need strides summing to 204, which forces 22/22.
        solution = planner.solve_overlaps(328, 124, 3, [False, False])
        self.assertEqual(solution.gaps, (22, 22))
        self.assertEqual(solution.matches, 0)

    def test_solve_overlaps_rejects_a_wrong_flag_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 3 boundaries"):
            planner.solve_overlaps(328, 124, 4, [False])


if __name__ == "__main__":
    unittest.main()
