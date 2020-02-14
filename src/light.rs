use crate::math::{f32x4, OrthonormalBasis, Vec3, Wec3};
use crate::spectrum::{Srgb, WSrgb};

pub trait Light: Send + Sync {
    // returns (sampled point, output radiance toward ref, pdf of sample wrt solid angle wrt ref point)
    fn sample(&self, samples: &[f32x4; 2], point: Wec3, normal: Wec3) -> (Wec3, WSrgb, f32x4);
}

#[derive(Clone, Copy)]
pub struct SphereLight {
    pos: Wec3,
    emission: WSrgb,
    rad: f32x4,
}

impl SphereLight {
    pub fn new(pos: Vec3, rad: f32, emission: Srgb) -> Self {
        Self {
            pos: Wec3::splat(pos),
            emission: WSrgb::splat(emission),
            rad: f32x4::from(rad),
        }
    }
}

impl Light for SphereLight {
    fn sample(&self, samples: &[f32x4; 2], p: Wec3, _n: Wec3) -> (Wec3, WSrgb, f32x4) {
        let dir = self.pos - p;
        let dist2 = dir.mag_sq();
        let dist = dist2.sqrt();
        let dir = dir / dist;
        let basis = (-dir).get_orthonormal_basis();

        let r2 = self.rad * self.rad;

        let sin_theta_max_2 = r2 / dist2;
        let cos_theta_max = f32x4::ZERO.max(f32x4::ONE - sin_theta_max_2).sqrt();
        let cos_theta = (f32x4::ONE - samples[0]) + samples[0] * cos_theta_max;
        let sin_theta = f32x4::ZERO.max(f32x4::ONE - cos_theta * cos_theta).sqrt();
        let phi = samples[1] * f32x4::TWO_PI;

        let ds = dist * cos_theta - f32x4::ZERO.max(r2 - dist2 * sin_theta * sin_theta).sqrt();
        let cos_alpha = (dist2 + r2 - ds * ds) / (f32x4::from(2.0) * dist * self.rad);
        let sin_alpha = f32x4::ZERO.max(f32x4::ONE - cos_alpha * cos_alpha).sqrt();

        let (sin_phi, cos_phi) = phi.sin_cos();

        let offset = basis.cols[0] * sin_alpha * cos_phi
            + basis.cols[1] * sin_alpha * sin_phi
            + basis.cols[2] * cos_alpha;

        let point = self.pos + offset * self.rad;

        let pdf = uniform_cone_pdf(cos_theta_max);

        (point, self.emission, pdf)
    }
}

fn uniform_cone_pdf(cos_theta_max: f32x4) -> f32x4 {
    f32x4::ONE / (f32x4::TWO_PI * (f32x4::ONE - cos_theta_max))
}
