use std::f32::consts::PI;

use rand::prelude::*;
use vek::vec::repr_c;

use color::Color;

pub type Vec3 = repr_c::Vec3<f32>;

pub trait RandomInit {
    fn rand(rng: &mut ThreadRng) -> Self;
}

impl RandomInit for Vec3 {
    fn rand(rng: &mut ThreadRng) -> Self {
        let theta = rng.gen_range::<f32>(0.0, 2.0 * PI);
        let phi = rng.gen_range::<f32>(-1.0, 1.0);
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
