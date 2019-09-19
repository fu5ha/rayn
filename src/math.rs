use std::f32::consts::PI;

use rand::prelude::*;
use vek::vec;

use crate::color::Color;

pub type Vec4 = vec::repr_c::Vec4<f32>;
pub type Vec3 = vec::repr_c::Vec3<f32>;
pub type Vec2 = vec::repr_c::Vec2<f32>;

pub type Quat = vek::quaternion::repr_c::Quaternion<f32>;

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub orientation: Quat,
}

pub trait RandomInit {
    fn rand(rng: &mut ThreadRng) -> Self;
}

impl RandomInit for Vec3 {
    fn rand(rng: &mut ThreadRng) -> Self {
        let theta = rng.gen_range(0f32, 2f32 * PI);
        let phi = rng.gen_range(-1f32, 1f32);
        let ophisq = (1.0 - phi * phi).sqrt();
        Vec3::new(ophisq * theta.cos(), ophisq * theta.sin(), phi)
    }
}

pub fn f0_from_ior(ior: f32) -> f32 {
    let f0 = (1.0 - ior) / (1.0 + ior);
    f0 * f0
}

pub fn f_schlick(cos: f32, f0: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos).powi(5)
}

pub fn f_schlick_c(cos: f32, f0: Color) -> Color {
    let f0 = f0.0;
    let out_v = f0 + (Vec3::from(1.0) - f0) * (1.0 - cos).powi(5);
    Color(out_v)
}

pub fn saturate(v: f32) -> f32 {
    v.min(1.0).max(0.0)
}
