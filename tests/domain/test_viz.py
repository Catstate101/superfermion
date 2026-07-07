"""Visualization domain tests."""

import pytest

import superfermion as sf


pytestmark = pytest.mark.domain

viz = pytest.importorskip("superfermion.viz", reason="viz module unavailable")


class TestDrawMpl:
    def test_draw_mpl_does_not_crash(self, bell_circuit):
        plt = pytest.importorskip("matplotlib.pyplot")
        fig = viz.draw_mpl(bell_circuit)
        assert fig is not None
        plt.close(fig)

    def test_draw_mpl_empty_circuit(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        c = sf.Circuit(2)
        fig = viz.draw_mpl(c)
        assert fig is not None
        plt.close(fig)

    def test_draw_mpl_parametric_gate(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        theta = sf.param("theta")
        c = sf.Circuit(1).ry(theta, 0)
        fig = viz.draw_mpl(c)
        assert fig is not None
        plt.close(fig)


class TestPlotHistogram:
    def test_plot_histogram_works(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        counts = {"00": 500, "01": 120, "10": 130, "11": 250}
        fig = viz.plot_histogram(counts, title="Bell Measurements")
        assert fig is not None
        plt.close(fig)

    def test_plot_histogram_sorted(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        counts = {"00": 100, "11": 900}
        fig = viz.plot_histogram(counts, sort="desc")
        assert fig is not None
        plt.close(fig)


