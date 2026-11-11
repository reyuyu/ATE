"""CPU-only checks for the additional nested reaccess scope and ranking rule."""
import unittest
from wider_reaccess import EXPANSION,expanded_grid,choose_validation


class WiderTests(unittest.TestCase):
    def test_nested_sets(self):
        values=list(EXPANSION.values())
        self.assertEqual([len(v) for v in values],[21,25,29,30])
        self.assertTrue(all(set(a)<set(b) for a,b in zip(values,values[1:])))

    def test_skip_stays_frozen(self):
        self.assertEqual(set(EXPANSION['Wide30_AllActive']),set(range(36))-set(range(1,7)))
        self.assertTrue(all(not set(v)&set(range(1,7)) for v in EXPANSION.values()))

    def test_budget_and_dedup(self):
        grid=expanded_grid()
        self.assertEqual(len(grid),41)
        self.assertEqual(len(set(tuple(v) for v in grid.values())),41)

    def test_validation_selection_no_heldout(self):
        rows=[dict(condition=n,k=k,all_tasks=True) for n,k in [('wide',30),('wider',29),('middle',25),('small',4),('eight',8),('other',2)]]
        self.assertEqual(choose_validation(rows),['wide','wider','middle','small','eight'])


if __name__=='__main__':
    unittest.main()
