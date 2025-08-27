import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class CarParams:
    """
    d_f, d_r : Represent grip of the car. Range: [0.015, 0.025]
    b_f, b_r: Slope of the pacejka. Range: [2.0 - 4.0].
    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.
    c_m_1: Motor parameter. Range [0.2, 0.5]
    c_m_1: Motor friction, Range [0.00, 0.007]
    c_f, c_r: [1.0 2.0] # motor parameters
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]
    """

    m: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(1.65): x  # type: ignore
    )  # [0.04, 0.08]
    i_com: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(2.78e-05): x  # type: ignore
    )  # [1e-6, 5e-6]
    l_f: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(0.13): x  # type: ignore
    )  # [0.025, 0.05]
    l_r: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(0.17): x  # type: ignore
    )  # [0.025, 0.05]
    g: jax.Array = dataclasses.field(default_factory=lambda x=jnp.array(9.81): x)  # type: ignore
    d_f: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(0.02): x  # type: ignore
    )  # [0.015, 0.025]
    c_f: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(1.2): x  # type: ignore
    )  # [1.0, 2.0]
    b_f: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(2.58): x  # type: ignore
    )  # [2.0, 4.0]
    d_r: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(0.017): x  # type: ignore
    )  # [0.015, 0.025]
    c_r: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(1.27): x  # type: ignore
    )  # [1.0, 2.0]
    b_r: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(3.39): x  # type: ignore
    )  # [2.0, 4.0]
    c_m_1: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(10.431917): x  # type: ignore
    )  # [0.2, 0.5]
    c_m_2: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(1.5003588): x  # type: ignore
    )  # [0.00, 0.007]
    c_d: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(0.0): x  # type: ignore
    )  # [0.01, 0.1]
    steering_limit: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array(0.19989373): x  # type: ignore
    )  # type: ignore
    use_blend: jax.Array = dataclasses.field(default_factory=lambda x=jnp.array(0.0): x)  # type: ignore
    # parameters used to compute the blend ratio characteristics
    blend_ratio_ub: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array([0.5477225575]): x  # type: ignore
    )
    blend_ratio_lb: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array([0.4472135955]): x  # type: ignore
    )
    angle_offset: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array([0.02791893]): x  # type: ignore
    )
    max_throttle: jax.Array = dataclasses.field(
        default_factory=lambda x=jnp.array([0.4]): x  # type: ignore
    )


def rotate_vector(v, theta):
    v_x, v_y = v[..., 0], v[..., 1]
    rot_x = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
    rot_y = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
    return jnp.concatenate([jnp.atleast_1d(rot_x), jnp.atleast_1d(rot_y)], axis=-1)


def compute_accelerations(x, u, params: CarParams):
    """Compute acceleration forces for dynamic model.
    Inputs
    -------
    x: jnp.ndarray,
        shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
    u: jnp.ndarray,
        shape = (2, ) -> [steering_angle, throttle]

    Output
    ------
    acceleration: jnp.ndarray,
        shape = (3, ) -> [a_r, a_t, a_theta]
    """
    i_com = params.i_com
    _, v_x, v_y, w = x[2], x[3], x[4], x[5]
    m = params.m
    l_f = params.l_f
    l_r = params.l_r
    d_f = params.d_f * params.g
    d_r = params.d_r * params.g
    c_f = params.c_f
    c_r = params.c_r
    b_f = params.b_f
    b_r = params.b_r
    c_m_1 = params.c_m_1
    c_m_2 = params.c_m_2
    c_d = params.c_d
    delta, d = u[0], u[1]
    alpha_f = -jnp.arctan((w * l_f + v_y) / (v_x + 1e-6)) + delta
    alpha_r = jnp.arctan((w * l_r - v_y) / (v_x + 1e-6))
    f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
    f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
    f_r_x = c_m_1 * d - (c_m_2**2) * v_x - (c_d**2) * (v_x * jnp.abs(v_x))
    v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
    v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
    w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r) / i_com
    acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
    return acceleration


class RaceCarDynamics:
    """
    local_coordinates: bool
        Used to indicate if local or global coordinates shall be used.
        If local, the state x is
            x = [0, 0, theta, vel_r, vel_t, angular_velocity_z]
        else:
            x = [x, y, theta, vel_x, vel_y, angular_velocity_z]
    u = [steering_angle, throttle]
    """

    def __init__(
        self,
        dt,
        local_coordinates: bool = False,
        rk_integrator: bool = True,
    ):
        if dt <= 1 / 100:
            integration_dt = dt
        else:
            integration_dt = 1 / 100
        self.local_coordinates = local_coordinates
        self.angle_idx = 2
        self.velocity_start_idx = 3
        self.velocity_end_idx = 4
        self.rk_integrator = rk_integrator
        self._num_steps_integrate = int(dt / integration_dt)
        self.dt_integration = integration_dt
        self.x_dim = 6
        self.u_dim = 2

    def _compute_one_dt(self, x, u, params: CarParams):
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, u, params)
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        theta = next_state[self.angle_idx]
        sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
        next_state = next_state.at[self.angle_idx].set(
            jnp.arctan2(sin_theta, cos_theta)
        )
        return next_state

    def rk_integration(
        self, x: jnp.array, u: jnp.array, params: CarParams
    ) -> jnp.array:
        integration_factors = jnp.asarray(
            [
                self.dt_integration / 2.0,
                self.dt_integration / 2.0,
                self.dt_integration,
                self.dt_integration,
            ]
        )
        integration_weights = jnp.asarray(
            [
                self.dt_integration / 6.0,
                self.dt_integration / 3.0,
                self.dt_integration / 3.0,
                self.dt_integration / 6.0,
            ]
        )

        def body(carry, _):
            """one step of rk integration.
            k_0 = self.ode(x, u)
            k_1 = self.ode(x + self.dt_integration / 2. * k_0, u)
            k_2 = self.ode(x + self.dt_integration / 2. * k_1, u)
            k_3 = self.ode(x + self.dt_integration * k_2, u)

            x_next = x + self.dt_integration * (k_0 / 6. + k_1 / 3. + k_2 / 3. + k_3 / 6.)
            """

            def rk_integrate(carry, ins):
                k = self.ode(carry, u, params)
                carry = carry + k * ins
                outs = k
                return carry, outs

            _, dxs = jax.lax.scan(rk_integrate, carry, xs=integration_factors, length=4)
            dx = (dxs.T * integration_weights).sum(axis=-1)
            q = carry + dx
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        theta = next_state[self.angle_idx]
        sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
        next_state = next_state.at[self.angle_idx].set(
            jnp.arctan2(sin_theta, cos_theta)
        )
        return next_state

    def step(self, x: jax.Array, u: jax.Array, params: Any) -> tuple[jax.Array, dict]:
        assert x.shape[-1] == 6
        theta_x = x[..., self.angle_idx]
        offset = jnp.clip(params.angle_offset, -jnp.pi, jnp.pi)
        theta_x = theta_x + offset
        if not self.local_coordinates:
            # rotate velocity to local frame to compute dx
            velocity_global = x[
                ..., self.velocity_start_idx : self.velocity_end_idx + 1
            ]
            rotated_vel = rotate_vector(velocity_global, -theta_x)
            x = x.at[..., self.velocity_start_idx : self.velocity_end_idx + 1].set(
                rotated_vel
            )
        if self.rk_integrator:
            next_x = self.rk_integration(x, u, params)
        else:
            next_x = self._compute_one_dt(x, u, params)
        if self.local_coordinates:
            # convert position to local frame
            pos = next_x[..., 0 : self.angle_idx] - x[..., 0 : self.angle_idx]
            rotated_pos = rotate_vector(pos, -theta_x)
            next_x = next_x.at[..., 0 : self.angle_idx].set(rotated_pos)
        else:
            # convert velocity to global frame
            new_theta_x = next_x[..., self.angle_idx]
            new_theta_x = new_theta_x + offset
            velocity = next_x[..., self.velocity_start_idx : self.velocity_end_idx + 1]
            rotated_vel = rotate_vector(velocity, new_theta_x)
            next_x = next_x.at[
                ..., self.velocity_start_idx : self.velocity_end_idx + 1
            ].set(rotated_vel)
        return next_x, {}

    def _ode_dyn(self, x, u, params: CarParams):
        """Compute derivative using dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        x_dot: jnp.ndarray,
            shape = (6, ) -> time derivative of x

        """
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle ant d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot])
        accelerations = compute_accelerations(x, u, params)
        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        """Compute kinematics derivative for localized state.
        Inputs
        -----
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, v_x, v_y, w], velocities in local frame
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        dx_kin: jnp.ndarray,
            shape = (6, ) -> derivative of x

        Assumption: \dot{\delta} = 0.
        """
        *_, theta, v_x, v_y, w = x[0], x[1], x[2], x[3], x[4], x[5]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        delta, d = u[0], u[1]
        v_r = v_x
        v_r_dot = (c_m_1 * d - (c_m_2**2) * v_r - (c_d**2) * (v_r * jnp.abs(v_r))) / m
        beta = jnp.arctan(jnp.tan(delta) * 1 / (l_r + l_f))
        v_x_dot = v_r_dot * jnp.cos(beta)
        # Determine accelerations from the kinematic model using FD.
        v_y_dot = (v_r * jnp.sin(beta) * l_r - v_y) / self.dt_integration
        # v_x_dot = (v_r_dot + v_y * w)
        # v_y_dot = - v_x * w
        w_dot = (jnp.sin(beta) * v_r - w) / self.dt_integration
        p_g_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_g_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        dx_kin = jnp.asarray([p_g_x_dot, p_g_y_dot, w, v_x_dot, v_y_dot, w_dot])
        return dx_kin

    def _compute_dx(self, x, u, params: CarParams):
        """Calculate time derivative of state.
        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x


        If params.use_blend <= 0.5 --> only kinematic model is used, else a blend between nonlinear model
        ant kinematic is used.
        """
        use_kin = params.use_blend <= 0.5
        v_x = x[3]
        blend_ratio_ub = jnp.square(params.blend_ratio_ub)
        blend_ratio_lb = jnp.square(params.blend_ratio_lb)
        blend_ratio = (v_x - blend_ratio_ub) / (blend_ratio_lb + 1e-6)
        blend_ratio = blend_ratio.squeeze()
        lambda_blend = jnp.min(jnp.asarray([jnp.max(jnp.asarray([blend_ratio, 0])), 1]))
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dx_dyn + (1 - lambda_blend) * dx_kin_full
        dx = (1 - use_kin) * dx_blend + use_kin * dx_kin_full
        return dx

    def ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/

        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x
        """
        assert x.shape[-1] == self.x_dim and u.shape[-1] == self.u_dim
        delta, d = u[0], u[1]
        delta = jnp.clip(delta, a_min=-1, a_max=1) * params.steering_limit
        d = jnp.clip(d, a_min=-1.0, a_max=1)  # throttle
        u = u.at[0].set(delta)
        u = u.at[1].set(d)
        dx = self._compute_dx(x, u, params)
        return dx
