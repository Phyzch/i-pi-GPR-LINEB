"""Unit tests for GPRConfig, MAPNEBGPRMover, and SharedData in neb_instanton_gpr.py."""

import numpy as np
import pytest

from ipi.engine.motion.neb_instanton_gpr import GPRConfig, MAPNEBGPRMover, SharedData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mover(**kwargs):
    """Return a minimal MAPNEBGPRMover.

    Only gpr_kernel_outputscale and gpr_kernel_lengthscale_ratio are required
    (the constructor raises when they are empty).  All other parameters use
    their documented defaults unless overridden via kwargs.
    """
    defaults = dict(
        gpr_kernel_outputscale=np.ones(1),
        gpr_kernel_lengthscale_ratio=np.ones(1),
        gpr_SE_kernel_number=1,
    )
    defaults.update(kwargs)
    return MAPNEBGPRMover(**defaults)


def full_gpr_config(**overrides) -> GPRConfig:
    """Return a GPRConfig with all fields filled in sensible defaults."""
    fields = dict(
        energy_shift=0.0,
        fix_dofs=np.array([]),
        gpr_SE_kernel_number=1,
        gpr_kernel_outputscale=np.ones(1),
        gpr_kernel_outputscale_constraint={"min": 0.0, "max": 1.0},
        gpr_kernel_lengthscale_ratio=np.ones(1),
        gpr_kernel_lengthscale_ratio_constraint={"min": 0.5, "max": 5.0},
        gpr_noise_std={"pot_noise_prior": 1e-6, "force_noise_prior": 1e-4, "hessian_noise_prior": 1e-3},
        gpr_fix_internal_dofs_bool=True,
        gpr_fix_internal_dofs_cutoff=1e-3,
        gpr_rigid_internal_dofs_cutoff=1e-2,
        gpr_covar_inverse_nugget=1e-8,
        gpr_trust_region=0.1,
        distance_cutoff_for_training_data=0.01,
        gpr_force_uncertainty_criterion=0.001,
        gpr_relative_force_error_criterion=0.05,
        gpr_absolute_force_error_criterion=0.001,
        read_initial_gpr_training_data=False,
        read_gpr_hessian_folder="None",
        test_gpr_model_along_instanton_path=False,
        ab_initio_hessian_bool=False,
        Hessian_interpolation="GPR",
        add_new_hessian_data_bool=False,
        candidate_hessian_data_number=20,
        new_hessian_data_index=np.zeros(0, int),
        add_new_grad_data_bool=False,
        candidate_grad_data_number=100,
        new_grad_data_index=np.zeros(0, int),
        train_grad_model_bool=True,
        train_hessian_model_bool=True,
        selective_hessian_bool=False,
        new_hessian_data_index_rigid_mode=np.zeros(0, int),
        cross_validation_bool=False,
        ridge_regularization_alpha={"force": 0.1, "hessian": 0.5},
    )
    fields.update(overrides)
    return GPRConfig(**fields)


# ---------------------------------------------------------------------------
# GPRConfig dataclass
# ---------------------------------------------------------------------------

class TestGPRConfig:
    def test_construction_and_scalar_fields(self):
        cfg = full_gpr_config(energy_shift=0.5, gpr_SE_kernel_number=2,
                              gpr_kernel_outputscale=np.ones(2),
                              gpr_kernel_lengthscale_ratio=np.ones(2))
        assert cfg.energy_shift == 0.5
        assert cfg.gpr_SE_kernel_number == 2

    def test_bool_fields(self):
        cfg = full_gpr_config(
            gpr_fix_internal_dofs_bool=True,
            ab_initio_hessian_bool=False,
            selective_hessian_bool=True,
            cross_validation_bool=False,
            read_initial_gpr_training_data=True,
        )
        assert cfg.gpr_fix_internal_dofs_bool is True
        assert cfg.ab_initio_hessian_bool is False
        assert cfg.selective_hessian_bool is True
        assert cfg.cross_validation_bool is False
        assert cfg.read_initial_gpr_training_data is True

    def test_string_fields(self):
        cfg = full_gpr_config(
            Hessian_interpolation="CubicSpline",
            read_gpr_hessian_folder="some/path",
        )
        assert cfg.Hessian_interpolation == "CubicSpline"
        assert cfg.read_gpr_hessian_folder == "some/path"

    def test_array_fields(self):
        idx = np.array([1, 3, 5])
        fix = np.array([0, 6])
        cfg = full_gpr_config(new_hessian_data_index=idx, fix_dofs=fix)
        np.testing.assert_array_equal(cfg.new_hessian_data_index, idx)
        np.testing.assert_array_equal(cfg.fix_dofs, fix)

    def test_dict_fields(self):
        noise = {"pot_noise_prior": 1e-5, "force_noise_prior": 1e-3, "hessian_noise_prior": 1e-2}
        alpha = {"force": 0.3, "hessian": 0.7}
        cfg = full_gpr_config(gpr_noise_std=noise, ridge_regularization_alpha=alpha)
        assert cfg.gpr_noise_std == noise
        assert cfg.ridge_regularization_alpha == alpha

    def test_no_stage_field(self):
        """stage is mutable runtime state on MAPNEBGPRMover, not part of GPRConfig."""
        cfg = full_gpr_config()
        assert not hasattr(cfg, "stage")


# ---------------------------------------------------------------------------
# MAPNEBGPRMover initialisation
# ---------------------------------------------------------------------------

class TestMAPNEBGPRMoverInit:
    def test_default_options(self):
        mover = make_mover()
        assert mover.options["stage"] == "neb"
        assert mover.options["opt"] == "neb"
        assert mover.options["mode"] == "verlet"

    def test_default_optarrays(self):
        mover = make_mover()
        assert mover.optarrays["energy_shift"] == 0.0

    def test_custom_stage(self):
        assert make_mover(stage="instanton").options["stage"] == "instanton"

    def test_custom_energy_shift(self):
        assert make_mover(energy_shift=1.5).optarrays["energy_shift"] == 1.5

    def test_string_optimizer(self):
        mover = make_mover(opt="string")
        assert mover.options["opt"] == "string"

    def test_multiple_kernels(self):
        mover = make_mover(
            gpr_SE_kernel_number=3,
            gpr_kernel_outputscale=np.array([1.0, 2.0, 3.0]),
            gpr_kernel_lengthscale_ratio=np.array([0.5, 1.0, 1.5]),
        )
        assert mover.options["gpr_SE_kernel_number"] == 3
        assert len(mover.optarrays["gpr_kernel_outputscale"]) == 3

    def test_distance_cutoff_clamped_to_trust_region(self):
        """When distance_cutoff > trust_region it is clamped to trust_region."""
        mover = make_mover(gpr_trust_region=0.05, distance_cutoff_for_training_data=0.2)
        assert mover.options["distance_cutoff_for_training_data"] == pytest.approx(0.05)

    def test_distance_cutoff_unchanged_when_smaller(self):
        mover = make_mover(gpr_trust_region=0.2, distance_cutoff_for_training_data=0.05)
        assert mover.options["distance_cutoff_for_training_data"] == pytest.approx(0.05)

    def test_distance_cutoff_unchanged_when_equal(self):
        mover = make_mover(gpr_trust_region=0.1, distance_cutoff_for_training_data=0.1)
        assert mover.options["distance_cutoff_for_training_data"] == pytest.approx(0.1)

    def test_kernel_lengthscale_count_mismatch_raises(self):
        with pytest.raises(AssertionError):
            MAPNEBGPRMover(
                gpr_SE_kernel_number=2,
                gpr_kernel_outputscale=np.ones(2),
                gpr_kernel_lengthscale_ratio=np.ones(1),   # should be 2
            )

    def test_kernel_outputscale_count_mismatch_raises(self):
        with pytest.raises(AssertionError):
            MAPNEBGPRMover(
                gpr_SE_kernel_number=2,
                gpr_kernel_outputscale=np.ones(1),          # should be 2
                gpr_kernel_lengthscale_ratio=np.ones(2),
            )

    def test_invalid_opt_raises(self):
        with pytest.raises(ValueError):
            make_mover(opt="gradient_descent")

    def test_shared_data_reset_on_init(self):
        make_mover()
        assert SharedData.ab_initio_bead_calculation_number == 0
        assert SharedData.inner_loop_optimization_step == 0

    def test_custom_tolerances_stored(self):
        tol = {"gradient": 1e-4, "gradient_end_bead": 2e-3,
               "action_forces_sum": 1e-4, "action": 1e-5}
        mover = make_mover(tolerances=tol)
        assert mover.options["tolerances"]["gradient"] == pytest.approx(1e-4)

    def test_fix_dofs_stored(self):
        fix = np.array([0, 3, 6])
        mover = make_mover(fix_dofs=fix)
        np.testing.assert_array_equal(mover.optarrays["fix_dofs"], fix)


# ---------------------------------------------------------------------------
# stage property
# ---------------------------------------------------------------------------

class TestStageProperty:
    def test_getter_reads_options_dict(self):
        mover = make_mover(stage="neb")
        assert mover.stage == "neb"

    def test_getter_reflects_direct_options_mutation(self):
        mover = make_mover()
        mover.options["stage"] = "instanton"
        assert mover.stage == "instanton"

    def test_setter_updates_options_dict(self):
        mover = make_mover(stage="neb")
        mover.stage = "instanton"
        assert mover.options["stage"] == "instanton"

    def test_setter_to_converged(self):
        mover = make_mover(stage="instanton")
        mover.stage = "converged"
        assert mover.stage == "converged"
        assert mover.options["stage"] == "converged"

    def test_stage_roundtrip_all_values(self):
        mover = make_mover()
        for stage in ("neb", "instanton", "converged"):
            mover.stage = stage
            assert mover.stage == stage


# ---------------------------------------------------------------------------
# gpr_config property
# ---------------------------------------------------------------------------

class TestGPRConfigProperty:
    def test_returns_gpr_config_instance(self):
        assert isinstance(make_mover().gpr_config, GPRConfig)

    def test_each_call_returns_fresh_instance(self):
        mover = make_mover()
        assert mover.gpr_config is not mover.gpr_config

    def test_energy_shift(self):
        assert make_mover(energy_shift=2.5).gpr_config.energy_shift == pytest.approx(2.5)

    def test_gpr_se_kernel_number(self):
        assert make_mover(gpr_SE_kernel_number=1).gpr_config.gpr_SE_kernel_number == 1

    def test_kernel_outputscale(self):
        scale = np.array([0.5, 1.5])
        mover = make_mover(gpr_SE_kernel_number=2,
                           gpr_kernel_outputscale=scale,
                           gpr_kernel_lengthscale_ratio=np.ones(2))
        np.testing.assert_array_equal(mover.gpr_config.gpr_kernel_outputscale, scale)

    def test_kernel_lengthscale_ratio(self):
        ratio = np.array([0.8, 1.2])
        mover = make_mover(gpr_SE_kernel_number=2,
                           gpr_kernel_outputscale=np.ones(2),
                           gpr_kernel_lengthscale_ratio=ratio)
        np.testing.assert_array_equal(mover.gpr_config.gpr_kernel_lengthscale_ratio, ratio)

    def test_gpr_trust_region(self):
        assert make_mover(gpr_trust_region=0.25).gpr_config.gpr_trust_region == pytest.approx(0.25)

    def test_fix_dofs(self):
        fix = np.array([0, 3, 6])
        np.testing.assert_array_equal(make_mover(fix_dofs=fix).gpr_config.fix_dofs, fix)

    def test_noise_std(self):
        noise = {"pot_noise_prior": 1e-5, "force_noise_prior": 1e-3, "hessian_noise_prior": 1e-2}
        assert make_mover(gpr_noise_std=noise).gpr_config.gpr_noise_std == noise

    def test_hessian_interpolation(self):
        assert make_mover(Hessian_interpolation="CubicSpline").gpr_config.Hessian_interpolation == "CubicSpline"

    def test_ab_initio_hessian_bool(self):
        assert make_mover(ab_initio_hessian_bool=True).gpr_config.ab_initio_hessian_bool is True

    def test_read_initial_gpr_training_data(self):
        assert make_mover(read_initial_gpr_training_data=True).gpr_config.read_initial_gpr_training_data is True

    def test_new_hessian_data_index(self):
        idx = np.array([2, 5, 7])
        np.testing.assert_array_equal(make_mover(new_hessian_data_index=idx).gpr_config.new_hessian_data_index, idx)

    def test_ridge_regularization_alpha(self):
        alpha = {"force": 0.3, "hessian": 0.7}
        assert make_mover(ridge_regularization_alpha=alpha).gpr_config.ridge_regularization_alpha == alpha

    def test_reflects_live_optarrays_mutation(self):
        """gpr_config is computed fresh on each access — reflects live optarrays state."""
        mover = make_mover(gpr_trust_region=0.1)
        mover.optarrays["gpr_trust_region"] = 0.99
        assert mover.gpr_config.gpr_trust_region == pytest.approx(0.99)

    def test_reflects_live_options_mutation(self):
        mover = make_mover(Hessian_interpolation="GPR")
        mover.options["Hessian_interpolation"] = "CubicSpline"
        assert mover.gpr_config.Hessian_interpolation == "CubicSpline"

    def test_no_stage_in_config(self):
        assert not hasattr(make_mover().gpr_config, "stage")

    def test_covar_inverse_nugget(self):
        assert make_mover(gpr_covar_inverse_nugget=1e-6).gpr_config.gpr_covar_inverse_nugget == pytest.approx(1e-6)

    def test_selective_hessian_bool(self):
        assert make_mover(selective_hessian_bool=True).gpr_config.selective_hessian_bool is True

    def test_cross_validation_bool(self):
        assert make_mover(cross_validation_bool=True).gpr_config.cross_validation_bool is True

    def test_train_grad_model_bool(self):
        assert make_mover(train_grad_model_bool=False).gpr_config.train_grad_model_bool is False

    def test_candidate_hessian_data_number(self):
        assert make_mover(candidate_hessian_data_number=30).gpr_config.candidate_hessian_data_number == 30


# ---------------------------------------------------------------------------
# SharedData class variables
# ---------------------------------------------------------------------------

class TestSharedData:
    def setup_method(self):
        SharedData.ab_initio_bead_calculation_number = 0
        SharedData.inner_loop_optimization_step = 0

    def test_initial_values(self):
        assert SharedData.ab_initio_bead_calculation_number == 0
        assert SharedData.inner_loop_optimization_step == 0

    def test_increment_ab_initio_count(self):
        SharedData.ab_initio_bead_calculation_number += 3
        assert SharedData.ab_initio_bead_calculation_number == 3

    def test_increment_inner_loop_step(self):
        SharedData.inner_loop_optimization_step += 10
        assert SharedData.inner_loop_optimization_step == 10

    def test_class_variable_shared_across_accesses(self):
        SharedData.ab_initio_bead_calculation_number = 99
        assert SharedData().ab_initio_bead_calculation_number == 99

    def test_mover_init_resets_count(self):
        SharedData.ab_initio_bead_calculation_number = 50
        make_mover()
        assert SharedData.ab_initio_bead_calculation_number == 0
