use bumpalo::Bump;

use rand::rngs::SmallRng;
use rand::Rng;

use arrayref::array_ref;

use crate::hitable::WShadingPoint;
use crate::math::{f32x4, f_schlick, f_schlick_c, OrthonormalBasis, RandomSample3d, Wec3};
use crate::spectrum::WSrgb;

use std::f32::consts::PI;

pub trait BSDF {
    fn scatter(
        &self,
        wo: Wec3,
        intersection: &WShadingPoint,
        samples: &[f32x4; 5],
    ) -> Option<WScatteringEvent>;

    fn le(&self, _wo: Wec3, _intersection: &WShadingPoint) -> WSrgb {
        WSrgb::zero()
    }
}

pub trait Material: Send + Sync {
    #[allow(clippy::mut_from_ref)]
    fn get_bsdf_at<'bump>(
        &self,
        intersection: &WShadingPoint,
        bump: &'bump Bump,
    ) -> &'bump mut dyn BSDF;
}

pub struct WScatteringEvent {
    pub wi: Wec3,
    pub f: WSrgb,
    pub pdf: f32x4,
    pub specular: f32x4,
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialHandle(pub usize);

pub struct MaterialStore(Vec<Box<dyn Material>>);

impl MaterialStore {
    pub fn new() -> Self {
        MaterialStore(Vec::new())
    }

    pub fn add_material<M: Material + 'static>(&mut self, material: M) -> MaterialHandle {
        self.0.push(Box::new(material));
        MaterialHandle(self.0.len() - 1)
    }

    pub fn get(&self, handle: MaterialHandle) -> &dyn Material {
        self.0[handle.0].as_ref()
    }
}

pub trait WShadingParamGenerator<T> {
    fn gen(&self, intersection: &WShadingPoint) -> T;
}

impl<T, I: Into<T> + Copy> WShadingParamGenerator<T> for I {
    fn gen(&self, _intersection: &WShadingPoint) -> T {
        (*self).into()
    }
}

#[derive(Clone, Copy)]
pub struct LambertianBSDF {
    albedo: WSrgb,
}

#[allow(dead_code)]
pub struct Lambertian<AG> {
    pub albedo_gen: AG,
}

impl<AG> Lambertian<AG> {
    #[allow(dead_code)]
    pub fn new(albedo_gen: AG) -> Self {
        Self { albedo_gen }
    }
}

impl<AG> Material for Lambertian<AG>
where
    AG: WShadingParamGenerator<WSrgb> + Send + Sync,
{
    fn get_bsdf_at<'bump>(
        &self,
        intersection: &WShadingPoint,
        bump: &'bump Bump,
    ) -> &'bump mut dyn BSDF {
        bump.alloc_with(|| LambertianBSDF {
            albedo: self.albedo_gen.gen(intersection),
        })
    }
}

impl BSDF for LambertianBSDF {
    fn scatter(
        &self,
        _wo: Wec3,
        intersection: &WShadingPoint,
        samples: &[f32x4; 5],
    ) -> Option<WScatteringEvent> {
        let diffuse_sample = Wec3::cosine_weighted_in_hemisphere(array_ref![samples,0,2], f32x4::ONE);
        let diffuse_bounce = (intersection.basis * diffuse_sample).normalized();
        let diffuse_pdf = diffuse_sample.dot(Wec3::unit_z()) / f32x4::from(PI);
        let diffuse_f = self.albedo / f32x4::from(PI);

        Some(WScatteringEvent {
            wi: diffuse_bounce,
            f: diffuse_f,
            pdf: diffuse_pdf,
            specular: f32x4::ZERO,
        })
    }
}

#[derive(Clone, Copy)]
pub struct DielectricBSDF {
    albedo: WSrgb,
    roughness: f32x4,
}

pub struct Dielectric<AG, RG> {
    pub albedo_gen: AG,
    pub roughness_gen: RG,
}

impl<AG, RG> Dielectric<AG, RG> {
    pub fn new(albedo_gen: AG, roughness_gen: RG) -> Self {
        Self {
            albedo_gen,
            roughness_gen,
        }
    }
}

impl<AG, RG> Material for Dielectric<AG, RG>
where
    AG: WShadingParamGenerator<WSrgb> + Send + Sync,
    RG: WShadingParamGenerator<f32x4> + Send + Sync,
{
    fn get_bsdf_at<'bump>(
        &self,
        intersection: &WShadingPoint,
        bump: &'bump Bump,
    ) -> &'bump mut dyn BSDF {
        bump.alloc_with(|| DielectricBSDF {
            albedo: self.albedo_gen.gen(intersection),
            roughness: self.roughness_gen.gen(intersection),
        })
    }
}

impl BSDF for DielectricBSDF {
    fn scatter(
        &self,
        wo: Wec3,
        intersection: &WShadingPoint,
        samples: &[f32x4; 5],
    ) -> Option<WScatteringEvent> {
        let norm = intersection.normal;
        let cos = norm.dot(-wo).abs();

        // diffuse part
        let diffuse_sample = Wec3::cosine_weighted_in_hemisphere(array_ref![samples,0,2], f32x4::ONE);
        let diffuse_bounce = (intersection.basis * diffuse_sample).normalized();
        let diffuse_pdf = diffuse_sample.dot(Wec3::unit_z()) / f32x4::from(PI);
        let diffuse_f = self.albedo / f32x4::from(PI);

        // spec part
        let spec_sample = Wec3::cosine_weighted_in_hemisphere(array_ref![samples,0,2], self.roughness);
        let reflection = wo.reflected(norm);
        let basis = reflection.get_orthonormal_basis();
        let spec_bounce = (basis * spec_sample).normalized();
        let spec_pdf = spec_sample.dot(Wec3::unit_z()) / f32x4::from(PI);
        let spec_f = WSrgb::one() / spec_bounce.dot(norm).abs() / f32x4::from(PI);

        // merge by fresnel
        let fresnel = f_schlick(cos, f32x4::from(0.04));

        let fresnel_sample = f32x4::from(samples[4]);

        let fresnel_mask = fresnel_sample.cmp_lt(fresnel);

        Some(WScatteringEvent {
            wi: Wec3::merge(fresnel_mask, spec_bounce, diffuse_bounce),
            f: WSrgb::merge(fresnel_mask, spec_f, diffuse_f),
            pdf: f32x4::merge(fresnel_mask, spec_pdf, diffuse_pdf),
            specular: f32x4::merge(fresnel_mask, f32x4::ONE, f32x4::ZERO),
        })
    }
}

#[allow(dead_code)]
pub struct Metallic<FG, RG> {
    pub f0_gen: FG,
    pub roughness_gen: RG,
}

impl<FG, RG> Metallic<FG, RG> {
    #[allow(dead_code)]
    pub fn new(f0_gen: FG, roughness_gen: RG) -> Self {
        Self {
            f0_gen,
            roughness_gen,
        }
    }
}

impl<AG, RG> Material for Metallic<AG, RG>
where
    AG: WShadingParamGenerator<WSrgb> + Send + Sync,
    RG: WShadingParamGenerator<f32x4> + Send + Sync,
{
    fn get_bsdf_at<'bump>(
        &self,
        intersection: &WShadingPoint,
        bump: &'bump Bump,
    ) -> &'bump mut dyn BSDF {
        bump.alloc_with(|| MetallicBSDF {
            f0: self.f0_gen.gen(intersection),
            roughness: self.roughness_gen.gen(intersection),
        })
    }
}

#[derive(Clone, Copy)]
pub struct MetallicBSDF {
    f0: WSrgb,
    roughness: f32x4,
}

impl BSDF for MetallicBSDF {
    // samples must contain at least two samples.
    fn scatter(
        &self,
        wo: Wec3,
        intersection: &WShadingPoint,
        samples: &[f32x4; 5],
    ) -> Option<WScatteringEvent> {
        let sample = unsafe { Wec3::cosine_weighted_in_hemisphere(array_ref![samples,0,2], self.roughness) };
        let reflection = wo.reflected(intersection.normal);
        let basis = reflection.get_orthonormal_basis();
        let bounce = basis * sample;
        let pdf = sample.dot(Wec3::unit_z()) / f32x4::from(PI);
        let cos = bounce.dot(intersection.normal).abs();
        let f = f_schlick_c(cos, self.f0) / cos / f32x4::from(PI);
        Some(WScatteringEvent {
            wi: bounce.normalized(),
            f,
            pdf,
            specular: f32x4::ONE,
        })
    }
}

// #[derive(Clone, Copy)]
// pub struct Refractive<S> {
//     refract_color: S,
//     ior: f32,
//     roughness: f32,
// }

// impl<S> Refractive<S> {
//     pub fn new(refract_color: S, roughness: f32, ior: f32) -> Self {
//         Refractive {
//             refract_color,
//             roughness,
//             ior,
//         }
//     }
// }

// impl BSDF for Refractive {
//     fn scatter(
//         &self,
//         wo: Wec3,
//         intersection: &mut WShadingPoint,
//         rng: &mut SmallRng,
//     ) -> WScatteringEvent {
//         let norm = intersection.normal;
//         let odn = wo.dot(norm);
//         let (refract_norm, eta, cos) = if odn > 0.0 {
//             (norm * -1.0, self.ior, odn)
//         } else {
//             (norm, 1.0 / self.ior, -odn)
//         };
//         let f0 = f0_from_ior(self.ior);
//         let fresnel = f_schlick(saturate(cos), f0);

//         let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);

//         let (f, pdf, bounce) = if rng.gen::<f32>() > fresnel {
//             let refraction = wo.refracted(refract_norm, eta);
//             if refraction != Vec3::zero() {
//                 let basis = refraction.get_orthonormal_basis();
//                 let bounce = basis * sample;
//                 let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
//                 let f = self.refract_color / bounce.dot(norm).abs() / std::f32::consts::PI;
//                 (f, pdf, bounce)
//             } else {
//                 // Total internal reflection
//                 reflect_part(wo, sample, norm)
//             }
//         } else {
//             reflect_part(wo, sample, norm)
//         };

//         WScatteringEvent {
//             wi: bounce.normalized(),
//             f,
//             pdf,
//             specular: true,
//         }
//     }
// }

// fn reflect_part(wo: Wec3, sample: Wec3, norm: Wec3) -> (WSrgb, f32x4, Wec3) {
//     let reflection = wo.reflected(norm);
//     let basis = reflection.get_orthonormal_basis();
//     let bounce = basis * sample;
//     let pdf = sample.dot(Vec3::unit_z()) / wide::consts::PI;
//     let f = WSrgb::one() / bounce.dot(norm).abs() / wide::consts::PI;
//     (f, pdf, bounce)
// }

#[derive(Clone, Copy)]
pub struct Sky {}

impl Material for Sky {
    fn get_bsdf_at<'bump>(
        &self,
        _intersection: &WShadingPoint,
        bump: &'bump Bump,
    ) -> &'bump mut dyn BSDF {
        bump.alloc_with(|| SkyBSDF {})
    }
}

#[derive(Clone, Copy)]
pub struct SkyBSDF {}

impl BSDF for SkyBSDF {
    fn scatter(
        &self,
        _wo: Wec3,
        _intersection: &WShadingPoint,
        _samples: &[f32x4; 5],
    ) -> Option<WScatteringEvent> {
        None
    }

    fn le(&self, wo: Wec3, _intersection: &WShadingPoint) -> WSrgb {
        let dir = -wo;
        let t = f32x4::from(0.5) * (dir.dot(Wec3::unit_y()) + f32x4::ONE);

        let bottom = WSrgb::new_splat(0.05, 0.025, 0.1);
        let top = WSrgb::new_splat(1.2, 1.15, 1.8);
        bottom * (f32x4::ONE - t) + top * t
    }
}

pub struct Emissive<EG> {
    pub emission_gen: EG,
}

impl<EG> Emissive<EG> {
    pub fn new(emission_gen: EG) -> Self {
        Self { emission_gen }
    }
}

impl<EG> Material for Emissive<EG>
where
    EG: WShadingParamGenerator<WSrgb> + Send + Sync,
{
    fn get_bsdf_at<'bump>(
        &self,
        intersection: &WShadingPoint,
        bump: &'bump Bump,
    ) -> &'bump mut dyn BSDF {
        bump.alloc_with(|| EmissiveBSDF {
            emission: self.emission_gen.gen(intersection),
            inner: LambertianBSDF {
                albedo: WSrgb::new_splat(0.5, 0.5, 0.5),
            },
        })
    }
}

#[derive(Clone, Copy)]
pub struct EmissiveBSDF<I> {
    inner: I,
    emission: WSrgb,
}

impl<I> BSDF for EmissiveBSDF<I>
where
    I: BSDF,
{
    fn scatter(
        &self,
        wo: Wec3,
        intersection: &WShadingPoint,
        samples: &[f32x4; 5],
    ) -> Option<WScatteringEvent> {
        self.inner.scatter(wo, intersection, samples)
    }

    fn le(&self, _wo: Wec3, _intersection: &WShadingPoint) -> WSrgb {
        self.emission
    }
}
