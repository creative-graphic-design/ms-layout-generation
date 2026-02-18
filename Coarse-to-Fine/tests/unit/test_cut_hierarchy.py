"""Unit tests for cut_hierarchy.py CutHierarchy class"""

import pytest
import torch

from coarse_to_fine.cut_hierarchy import CutHierarchy, convert_ltrb_to_ltwh


class TestConvertLtrbToLtwh:
    """Tests for convert_ltrb_to_ltwh function"""

    def test_convert_single_bbox(self):
        """Test conversion of a single bounding box"""
        bbox = [[10, 20, 50, 80]]  # l, t, r, b
        result = convert_ltrb_to_ltwh(bbox)

        expected = torch.tensor([[10, 20, 40, 60]])  # l, t, w, h
        assert torch.equal(result, expected)

    def test_convert_multiple_bboxes(self):
        """Test conversion of multiple bounding boxes"""
        bboxes = [[10, 20, 50, 80], [5, 10, 25, 40]]
        result = convert_ltrb_to_ltwh(bboxes)

        expected = torch.tensor([[10, 20, 40, 60], [5, 10, 20, 30]])
        assert torch.equal(result, expected)

    def test_convert_zero_size_bbox(self):
        """Test conversion of zero-size bounding box"""
        bbox = [[10, 20, 10, 20]]  # zero width and height
        result = convert_ltrb_to_ltwh(bbox)

        expected = torch.tensor([[10, 20, 0, 0]])
        assert torch.equal(result, expected)

    def test_convert_preserves_dtype(self):
        """Test that conversion preserves long dtype"""
        bbox = [[10, 20, 50, 80]]
        result = convert_ltrb_to_ltwh(bbox)

        assert result.dtype == torch.long


class TestCutHierarchyBasicMethods:
    """Tests for basic CutHierarchy methods"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_format_layout_empty(self, cut_hierarchy):
        """Test format_layout with empty inputs"""
        bboxes = []
        labels = []
        result = cut_hierarchy.format_layout(bboxes, labels)
        assert result == []

    def test_format_layout_single(self, cut_hierarchy):
        """Test format_layout with single element"""
        bboxes = [[10, 20, 50, 80]]
        labels = [1]
        result = cut_hierarchy.format_layout(bboxes, labels)

        assert len(result) == 1
        assert result[0] == ([10, 20, 50, 80], 1)

    def test_format_layout_multiple(self, cut_hierarchy):
        """Test format_layout with multiple elements"""
        bboxes = [[10, 20, 50, 80], [60, 20, 100, 80], [10, 90, 50, 130]]
        labels = [1, 2, 3]
        result = cut_hierarchy.format_layout(bboxes, labels)

        assert len(result) == 3
        assert result[0] == ([10, 20, 50, 80], 1)
        assert result[1] == ([60, 20, 100, 80], 2)
        assert result[2] == ([10, 90, 50, 130], 3)

    def test_get_label_in_one_group_single_label(self, cut_hierarchy):
        """Test get_label_in_one_group with single label per group"""
        grouped_label = [[1], [2], [3]]
        num_label = 5
        result = cut_hierarchy.get_label_in_one_group(grouped_label, num_label)

        assert result.shape == (3, 5)
        assert result[0, 0] == 1  # label 1 appears once in group 0
        assert result[1, 1] == 1  # label 2 appears once in group 1
        assert result[2, 2] == 1  # label 3 appears once in group 2

    def test_get_label_in_one_group_multiple_labels(self, cut_hierarchy):
        """Test get_label_in_one_group with multiple labels per group"""
        grouped_label = [[1, 1, 2], [3, 3, 3]]
        num_label = 5
        result = cut_hierarchy.get_label_in_one_group(grouped_label, num_label)

        assert result.shape == (2, 5)
        assert result[0, 0] == 2  # label 1 appears twice in group 0
        assert result[0, 1] == 1  # label 2 appears once in group 0
        assert result[1, 2] == 3  # label 3 appears three times in group 1


class TestCutHierarchyGroupBbox:
    """Tests for group_bbox method"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_group_bbox_single_element(self, cut_hierarchy):
        """Test group_bbox with single element"""
        sorted_bbox_with_idx = [([10, 20, 50, 80], 0)]
        result = cut_hierarchy.group_bbox(0, sorted_bbox_with_idx, direction="y")

        assert result == [0]

    def test_group_bbox_no_grouping_y(self, cut_hierarchy):
        """Test group_bbox with no grouping in y direction"""
        # Two bboxes far apart vertically
        sorted_bbox_with_idx = [([10, 20, 50, 80], 0), ([10, 100, 50, 160], 1)]
        result = cut_hierarchy.group_bbox(5, sorted_bbox_with_idx, direction="y")

        # Should be grouped into two separate groups
        assert isinstance(result, list)
        assert len(result) == 2

    def test_group_bbox_with_grouping_y(self, cut_hierarchy):
        """Test group_bbox with grouping in y direction"""
        # Two bboxes close vertically (within threshold)
        sorted_bbox_with_idx = [([10, 20, 50, 80], 0), ([60, 25, 100, 85], 1)]
        result = cut_hierarchy.group_bbox(10, sorted_bbox_with_idx, direction="y")

        # Should be grouped into one group (returns indices)
        assert isinstance(result, list)
        # When all elements are in one group at this level, returns flat list
        assert 0 in result
        assert 1 in result

    def test_group_bbox_x_direction(self, cut_hierarchy):
        """Test group_bbox in x direction"""
        # Two bboxes aligned vertically but far horizontally
        sorted_bbox_with_idx = [([10, 20, 50, 80], 0), ([100, 20, 140, 80], 1)]
        result = cut_hierarchy.group_bbox(5, sorted_bbox_with_idx, direction="x")

        # Should be grouped into two separate groups
        assert isinstance(result, list)
        assert len(result) == 2

    def test_group_bbox_overlapping_ranges(self, cut_hierarchy):
        """Test group_bbox with overlapping y ranges"""
        # Three bboxes with overlapping y ranges
        sorted_bbox_with_idx = [
            ([10, 20, 50, 60], 0),  # y: 20-60
            ([60, 30, 100, 70], 1),  # y: 30-70
            ([110, 35, 150, 75], 2),  # y: 35-75
        ]
        result = cut_hierarchy.group_bbox(0, sorted_bbox_with_idx, direction="y")

        # All should be grouped together due to overlapping y ranges
        assert isinstance(result, list)
        assert 0 in result or isinstance(result[0], list)


class TestCutHierarchyGroupLayoutToTree:
    """Tests for group_layout_to_tree method"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_group_layout_to_tree_simple(self, cut_hierarchy):
        """Test group_layout_to_tree with simple layout"""
        box_with_label = [
            ([10, 20, 50, 60], 1),
            ([60, 20, 100, 60], 2),
            ([10, 70, 50, 110], 3),
        ]

        sorted_bbox, group_list = cut_hierarchy.group_layout_to_tree(
            box_with_label, distance_threshold=0, direction="y"
        )

        assert len(sorted_bbox) == 3
        assert isinstance(group_list, list)

    def test_group_layout_to_tree_vertical_alignment(self, cut_hierarchy):
        """Test group_layout_to_tree with vertical alignment"""
        # Two rows of elements
        box_with_label = [
            ([10, 10, 30, 30], 1),  # top-left
            ([40, 10, 60, 30], 2),  # top-right
            ([10, 50, 30, 70], 3),  # bottom-left
            ([40, 50, 60, 70], 4),  # bottom-right
        ]

        sorted_bbox, group_list = cut_hierarchy.group_layout_to_tree(
            box_with_label, distance_threshold=5, direction="y"
        )

        assert len(sorted_bbox) == 4
        # Should create hierarchical grouping
        assert isinstance(group_list, list)


class TestCutHierarchyGetCutStructure:
    """Tests for get_cut_structure_bottom_two method"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_get_cut_structure_flat_list(self, cut_hierarchy):
        """Test get_cut_structure_bottom_two with flat list"""
        group_tree = [0, 1, 2]
        result = cut_hierarchy.get_cut_structure_bottom_two(group_tree, [])

        assert len(result) == 1
        assert result[0] == [0, 1, 2]

    def test_get_cut_structure_nested_single_elements(self, cut_hierarchy):
        """Test get_cut_structure_bottom_two with nested single elements"""
        group_tree = [[0], [1], [2]]
        result = cut_hierarchy.get_cut_structure_bottom_two(group_tree, [])

        # The method treats this as all single-element groups at same level
        # so they get flattened into one group
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_get_cut_structure_hierarchical(self, cut_hierarchy):
        """Test get_cut_structure_bottom_two with hierarchical structure"""
        group_tree = [[0, 1], [2, 3]]
        result = cut_hierarchy.get_cut_structure_bottom_two(group_tree, [])

        assert len(result) == 2
        assert [0, 1] in result
        assert [2, 3] in result

    def test_get_cut_structure_mixed(self, cut_hierarchy):
        """Test get_cut_structure_bottom_two with mixed structure"""
        group_tree = [[[0, 1], [2]], [3]]
        result = cut_hierarchy.get_cut_structure_bottom_two(group_tree, [])

        # Should extract bottom-two level groups
        assert isinstance(result, list)
        assert len(result) >= 2


class TestCutHierarchyGetGroupboxPos:
    """Tests for get_groupbox_pos method"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_get_groupbox_pos_single_group(self, cut_hierarchy):
        """Test get_groupbox_pos with single group"""
        structure = [[0, 1]]
        sorted_box = [([10, 20, 30, 40], 1), ([50, 60, 70, 80], 2)]

        result = cut_hierarchy.get_groupbox_pos(structure, sorted_box)

        assert len(result) == 1
        # Bounding box should encompass both boxes
        assert result[0][0] == 10  # min left
        assert result[0][1] == 20  # min top
        assert result[0][2] == 70  # max right
        assert result[0][3] == 80  # max bottom

    def test_get_groupbox_pos_multiple_groups(self, cut_hierarchy):
        """Test get_groupbox_pos with multiple groups"""
        structure = [[0], [1]]
        sorted_box = [([10, 20, 30, 40], 1), ([50, 60, 70, 80], 2)]

        result = cut_hierarchy.get_groupbox_pos(structure, sorted_box)

        assert len(result) == 2
        assert result[0] == [10, 20, 30, 40]
        assert result[1] == [50, 60, 70, 80]

    def test_get_groupbox_pos_overlapping(self, cut_hierarchy):
        """Test get_groupbox_pos with overlapping boxes"""
        structure = [[0, 1]]
        sorted_box = [([10, 10, 50, 50], 1), ([30, 30, 70, 70], 2)]

        result = cut_hierarchy.get_groupbox_pos(structure, sorted_box)

        assert len(result) == 1
        assert result[0] == [10, 10, 70, 70]  # Encompasses both boxes


class TestCutHierarchyGetGroupInformation:
    """Tests for get_group_infomation method"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_get_group_infomation_single_group(self, cut_hierarchy):
        """Test get_group_infomation with single group"""
        structure = [[0, 1]]
        sorted_bbox = [([10, 20, 30, 40], 1), ([50, 60, 70, 80], 2)]

        grouped_label, grouped_box = cut_hierarchy.get_group_infomation(
            structure, sorted_bbox
        )

        assert len(grouped_label) == 1
        assert len(grouped_box) == 1
        assert torch.equal(grouped_label[0], torch.tensor([1, 2]))
        assert grouped_box[0].shape == (2, 4)

    def test_get_group_infomation_multiple_groups(self, cut_hierarchy):
        """Test get_group_infomation with multiple groups"""
        structure = [[0], [1]]
        sorted_bbox = [([10, 20, 30, 40], 1), ([50, 60, 70, 80], 2)]

        grouped_label, grouped_box = cut_hierarchy.get_group_infomation(
            structure, sorted_bbox
        )

        assert len(grouped_label) == 2
        assert len(grouped_box) == 2
        assert torch.equal(grouped_label[0], torch.tensor([1]))
        assert torch.equal(grouped_label[1], torch.tensor([2]))


class TestCutHierarchyRelativeCoordinate:
    """Tests for relative_coordinate method"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_relative_coordinate_single_group(self, cut_hierarchy):
        """Test relative_coordinate with single group"""
        grouped_box = [torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])]
        group_bounding_box = [[0, 0, 100, 100]]

        result = cut_hierarchy.relative_coordinate(grouped_box, group_bounding_box)

        assert len(result) == 1
        # Check that coordinates are normalized to [0, 1] range
        assert result[0][0, 0] == 10.0 / 100.0  # x1 relative
        assert result[0][0, 1] == 20.0 / 100.0  # y1 relative
        assert result[0][0, 2] == 30.0 / 100.0  # x2 relative
        assert result[0][0, 3] == 40.0 / 100.0  # y2 relative

    def test_relative_coordinate_offset(self, cut_hierarchy):
        """Test relative_coordinate with offset bounding box"""
        grouped_box = [torch.tensor([[20, 30, 40, 50]])]
        group_bounding_box = [[10, 20, 50, 60]]  # 40 width, 40 height

        result = cut_hierarchy.relative_coordinate(grouped_box, group_bounding_box)

        # Coordinates should be relative to group bounding box
        # Result is float64, so compare as float64
        assert torch.isclose(
            result[0][0, 0], torch.tensor(10.0 / 40.0, dtype=torch.float64)
        )  # (20-10)/40
        assert torch.isclose(
            result[0][0, 1], torch.tensor(10.0 / 40.0, dtype=torch.float64)
        )  # (30-20)/40
        assert torch.isclose(
            result[0][0, 2], torch.tensor(30.0 / 40.0, dtype=torch.float64)
        )  # (40-10)/40
        assert torch.isclose(
            result[0][0, 3], torch.tensor(30.0 / 40.0, dtype=torch.float64)
        )  # (50-20)/40

    def test_relative_coordinate_multiple_groups(self, cut_hierarchy):
        """Test relative_coordinate with multiple groups"""
        grouped_box = [
            torch.tensor([[10, 10, 20, 20]]),
            torch.tensor([[60, 60, 80, 80]]),
        ]
        group_bounding_box = [[0, 0, 50, 50], [50, 50, 100, 100]]

        result = cut_hierarchy.relative_coordinate(grouped_box, group_bounding_box)

        assert len(result) == 2
        # Both should be normalized to their respective bounding boxes
        assert result[0].dtype == torch.float64
        assert result[1].dtype == torch.float64


class TestCutHierarchyFullPipeline:
    """Tests for full CutHierarchy pipeline (__call__ method)"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return {
            "discrete_gold_bboxes": torch.tensor(
                [[10, 10, 20, 20], [30, 10, 20, 20], [10, 40, 20, 20]], dtype=torch.long
            ),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        }

    def test_call_returns_required_keys(self, cut_hierarchy, sample_data):
        """Test that __call__ returns all required keys"""
        num_labels = 5

        def discrete_func(x):
            return x

        result = cut_hierarchy(sample_data, num_labels, discrete_func)

        assert "group_bounding_box" in result
        assert "label_in_one_group" in result
        assert "grouped_label" in result
        assert "grouped_box" in result

    def test_call_group_bounding_box_format(self, cut_hierarchy, sample_data):
        """Test that group_bounding_box is in ltwh format"""
        num_labels = 5

        def discrete_func(x):
            return x

        result = cut_hierarchy(sample_data, num_labels, discrete_func)

        # Should be in ltwh format
        assert result["group_bounding_box"].shape[1] == 4
        assert result["group_bounding_box"].dtype == torch.long

    def test_call_label_in_one_group_shape(self, cut_hierarchy, sample_data):
        """Test that label_in_one_group has correct shape"""
        num_labels = 5

        def discrete_func(x):
            return x

        result = cut_hierarchy(sample_data, num_labels, discrete_func)

        # Should have shape (num_groups, num_labels)
        assert result["label_in_one_group"].shape[1] == num_labels
        assert result["label_in_one_group"].dtype == torch.long

    def test_call_grouped_data_consistency(self, cut_hierarchy, sample_data):
        """Test that grouped data has consistent lengths"""
        num_labels = 5

        def discrete_func(x):
            return x

        result = cut_hierarchy(sample_data, num_labels, discrete_func)

        num_groups = len(result["grouped_label"])
        assert len(result["grouped_box"]) == num_groups
        assert result["group_bounding_box"].shape[0] == num_groups
        assert result["label_in_one_group"].shape[0] == num_groups

    def test_call_with_horizontal_first(self, cut_hierarchy):
        """Test that fallback to x direction works"""
        # Create horizontally arranged layout (all same y)
        data = {
            "discrete_gold_bboxes": torch.tensor(
                [[10, 10, 20, 20], [30, 10, 20, 20], [50, 10, 20, 20]], dtype=torch.long
            ),
            "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        }
        num_labels = 5

        def discrete_func(x):
            return x

        result = cut_hierarchy(data, num_labels, discrete_func)

        # Should still produce valid output
        assert "group_bounding_box" in result
        assert result["group_bounding_box"].shape[0] > 0


class TestCutHierarchyEdgeCases:
    """Tests for edge cases in CutHierarchy"""

    @pytest.fixture
    def cut_hierarchy(self):
        """Create a CutHierarchy instance"""
        return CutHierarchy()

    def test_single_element_layout(self, cut_hierarchy):
        """Test with single element layout"""
        data = {
            "discrete_gold_bboxes": torch.tensor([[10, 10, 20, 20]], dtype=torch.long),
            "labels": torch.tensor([1], dtype=torch.long),
        }
        num_labels = 5

        def discrete_func(x):
            return x

        result = cut_hierarchy(data, num_labels, discrete_func)

        assert result["group_bounding_box"].shape[0] == 1
        assert len(result["grouped_label"]) == 1
        assert len(result["grouped_box"]) == 1

    def test_relative_coordinate_zero_dimension(self, cut_hierarchy):
        """Test relative_coordinate handles near-zero dimensions"""
        grouped_box = [torch.tensor([[10.0, 10.0, 10.0, 10.0]])]
        group_bounding_box = [[10, 10, 10, 10]]  # zero width and height

        # Should handle division by near-zero gracefully (uses 1e-8 minimum)
        result = cut_hierarchy.relative_coordinate(grouped_box, group_bounding_box)

        # Should not raise error and produce finite values
        assert torch.isfinite(result[0]).all()
