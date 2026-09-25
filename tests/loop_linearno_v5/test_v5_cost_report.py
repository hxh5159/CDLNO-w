from tools.linearno_loop_v5_report import measurement_rows, parameter_rows

def test_cost_report_rows_and_bounded_cpu_measurement():
    rows = parameter_rows(tasks=("elasticity",), counts=(2, 3))
    assert len(rows) == 1 + 2 * (1 + 2 + 1)
    assert all(row["measured_parameters"] == row["analytic"]["parameters"] for row in rows)
    measured = measurement_rows(devices=["cpu"], warmup=1, steps=2)
    assert len(measured) == 5
    traced = next(row for row in measured if row["trace"] is not None)
    assert not traced["trace"]["forbidden_attention"]
    assert all(row["measurement"]["forward"]["median_ms"] > 0 for row in measured)
