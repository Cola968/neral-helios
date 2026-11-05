"""CPU-only contracts for SOLVE's rational intermediate-value boundary."""
import unittest

from solve import solve


class LinearBoundaryAudit(unittest.TestCase):
    def test_cross_side_subtraction_rejects_out_of_range_values(self):
        bound = 10**24
        cases = [
            f'{bound}*x=-{bound}*x',  # coefficient numerator
            f'2*x+{bound}=-{bound}',  # constant numerator, bounded solution
            'x/1000000000003=x/1000000000001',  # coefficient denominator
            'x/1000000000003+1/1000000000003=1/1000000000001',  # constant denominator
        ]
        for expression in cases:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, 'Zahlenbereich überschritten'):
                    solve(expression)

    def test_exact_boundary_and_existing_results_remain_supported(self):
        bound = 10**24
        cases = {
            f'{bound - 1}*x=-x': 'x = 0',
            f'2*x+{bound - 1}=-1': f'x = -{bound // 2}',
            f'x/{bound}=0': 'x = 0',
            f'x=1/{bound}': f'x = 1/{bound}',
            '2*x+3=11': 'x = 4',
            'x+1=x+1': 'Unendlich viele Lösungen',
            'x+1=x+2': 'Keine Lösung',
            '(-7)//3': '-3',
            '(-7)%3': '2',
            '2**-12': '1/4096',
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = solve(expression)
                self.assertEqual(result['result'], expected)
                self.assertFalse(result['model_generated'])


if __name__ == '__main__':
    unittest.main()
