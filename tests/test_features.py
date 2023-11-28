import pytest

from src.features.spatial import euclidean_distance, manhattan_distance


@pytest.mark.parametrize('distance', [euclidean_distance, manhattan_distance])
class TestDistanceRules:
    """
    Test that features satisfy distance rules.

    https://en.wikipedia.org/wiki/Distance#Mathematical_formalization
    """
    @pytest.mark.parametrize('x, y, identical', [
        ((0, 0), (0, 0), True),
        ((1, 1), (1, 1), True),
        ((-1, -1), (-1, -1), True),
        ((0, 0), (1, 1), False)
    ])
    def test_identity_of_indiscernibles(self, distance, x, y, identical):
        """Test that d(x, x) = 0 and that d(x, y) > 0 for x != y"""
        assert (distance(*x, *y) == 0) == identical

    @pytest.mark.parametrize('x, y', [
        ((0, 0), (1, 1)),
        ((-1, -1), (0, 0)),
        ((-1, -1), (1, 1))
    ])
    def test_symmetry(self, distance, x, y):
        """Test that d(x, y) = d(y, x)"""
        assert distance(*x, *y) == distance(*y, *x)

    @pytest.mark.parametrize('x, y, z', [
        ((0, 0), (2, 2), (-2, 2)),
        ((0, 0), (2, 1), (4, 2)),
    ], ids=["standard-triangle", "flat-triangle"])
    def test_triangle_inequality(self, distance, x, y, z):
        """Test that d(x, z) <= d(x, y) + d(y, z)"""
        assert distance(*x, *z) <= distance(*x, *y) + distance(*y, *z)


@pytest.mark.parametrize('distance, x, y, expected', [
    (euclidean_distance, (0, 0), (1, 1), 1.41),
    (euclidean_distance, (-2, -2), (2, 2), 5.66),
    (manhattan_distance, (0, 0), (1, 1), 2),
    (manhattan_distance, (-2, -2), (2, 2), 8),
])
def test_distance_result(distance, x, y, expected):
    """Test the results of distance features"""
    assert round(distance(*x, *y), 2) == expected
