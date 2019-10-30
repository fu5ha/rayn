use bumpalo::Bump;

use rand::rngs::SmallRng;
use rand::Rng;

use crate::hitable::{Intersection, WIntersection};
use crate::math::{
    f0_from_ior, f_schlick, f_schlick_c, saturate, OrthonormalBasis, RandomSample3d, Vec3, Wec3,
};
use crate::spectrum::{Srgb, WSrgb};

use wide::f32x4;

use std::any::Any;

pub trait BSDF {
    fn scatter(
        &self,
        wo: Wec3,
        intersection: &WIntersection,
        rng: &mut SmallRng,
    ) -> Option<WScatteringEvent>;

    fn le(&self, _wo: Wec3, _intersection: &WIntersection) -> WSrgb {
        WSrgb::zero()
    }
}

pub trait Material: Send + Sync {
    fn get_bsdf_at<'bump>(
        &self,
        intersection: &WIntersection,
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

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub trait WShadingParamGenerator<T> {
    fn gen(&self, intersection: &WIntersection) -> T;
}

impl<T, I: Into<T> + Copy> WShadingParamGenerator<T> for I {
    fn gen(&self, _intersection: &WIntersection) -> T {
        (*self).into()
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
        intersection: &WIntersection,
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
        intersection: &WIntersection,
        rng: &mut SmallRng,
    ) -> Option<WScatteringEvent> {
        let norm = intersection.normal;
        let cos = norm.dot(-wo).abs();

        // diffuse part
        let diffuse_sample = Wec3::cosine_weighted_in_hemisphere(rng, f32x4::from(1.0));
        let diffuse_bounce = (intersection.basis * diffuse_sample).normalized();
        let diffuse_pdf = diffuse_sample.dot(Wec3::unit_z()) / f32x4::from(wide::consts::PI);
        let diffuse_f = self.albedo / wide::consts::PI.into();

        // spec part
        let spec_sample = Wec3::cosine_weighted_in_hemisphere(rng, self.roughness);
        let reflection = wo.reflected(norm);
        let basis = reflection.get_orthonormal_basis();
        let spec_bounce = (basis * spec_sample).normalized();
        let spec_pdf = spec_sample.dot(Wec3::unit_z()) / f32x4::from(wide::consts::PI);
        let spec_f = WSrgb::one() / spec_bounce.dot(norm).abs() / f32x4::from(wide::consts::PI);

        // merge by fresnel
        let fresnel = f_schlick(cos, f32x4::from(0.04));

        let fresnel_sample = f32x4::new(
            rng.gen::<f32>(),
            rng.gen::<f32>(),
            rng.gen::<f32>(),
            rng.gen::<f32>(),
        );

        let fresnel_mask = fresnel_sample.cmp_lt(fresnel);

        Some(WScatteringEvent {
            wi: spec_bounce,
            f: spec_f,
            pdf: spec_pdf,
            specular: f32x4::from(1.0),
            // wi: Wec3::merge(fresnel_mask, diffuse_bounce, spec_bounce),
            // f: WSrgb::merge(fresnel_mask, diffuse_f, spec_f),
            // pdf: f32x4::merge(fresnel_mask, diffuse_pdf, spec_pdf),
            // specular: f32x4::merge(fresnel_mask, f32x4::from(0.0), f32x4::from(1.0)),
        })
    }
}

#[derive(Clone, Copy)]
pub struct MetalBSDF {
    f0: WSrgb,
    roughness: f32x4,
}

impl BSDF for MetalBSDF {
    fn scatter(
        &self,
        wo: Wec3,
        intersection: &WIntersection,
        rng: &mut SmallRng,
    ) -> Option<WScatteringEvent> {
        let sample = Wec3::cosine_weighted_in_hemisphere(rng, self.roughness);
        let reflection = wo.reflected(intersection.normal);
        let basis = reflection.get_orthonormal_basis();
        let bounce = basis * sample;
        let pdf = sample.dot(Wec3::unit_z()) / f32x4::from(wide::consts::PI);
        let cos = bounce.dot(intersection.normal).abs();
        let f = f_schlick_c(cos, self.f0) / cos / f32x4::from(wide::consts::PI);
        Some(WScatteringEvent {
            wi: bounce.normalized(),
            f,
            pdf,
            specular: f32x4::from(1.0),
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
//         intersection: &mut WIntersection,
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
        _intersection: &WIntersection,
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
        _intersection: &WIntersection,
        _rng: &mut SmallRng,
    ) -> Option<WScatteringEvent> {
        None
    }

    fn le(&self, wo: Wec3, _intersection: &WIntersection) -> WSrgb {
        let dir = -wo;
        let t = f32x4::from(0.5) * (dir.y + f32x4::from(1.0));

        let top = WSrgb::one();
        let mid = WSrgb::new(f32x4::from(0.5), f32x4::from(0.7), f32x4::from(1.0));
        top * (f32x4::from(1.0) - t) + mid * t
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
        intersection: &WIntersection,
        rng: &mut SmallRng,
    ) -> Option<WScatteringEvent> {
        self.inner.scatter(wo, intersection, rng)
    }

    fn le(&self, _wo: Wec3, _intersection: &WIntersection) -> WSrgb {
        self.emission
    }
}

// #[derive(Copy, Clone)]
// pub struct Checkerboard3d<M1, M2> {
//     pub mat1: M1,
//     pub mat2: M2,
//     pub scale: Vec3,
// }

// impl<M1, M2> Checkerboard3d<M1, M2> {
//     pub fn new(scale: Vec3, mat1: M1, mat2: M2) -> Self {
//         Checkerboard3d { scale, mat1, mat2 }
//     }
// }

// trait ModuloSigned {
//     fn modulo(&self, n: Self) -> Self;
// }

// impl<T> ModuloSigned for T
// where
//     T: std::ops::Add<Output = T> + std::ops::Rem<Output = T> + Copy,
// {
//     fn modulo(&self, n: T) -> T {
//         (*self % n + n) % n
//     }
// }

// impl<S, M1, M2> Material<S> for Checkerboard3d<M1, M2>
// where
//     S: IsSpectrum,
//     M1: Material<S>,
//     M2: Material<S>,
// {
//     fn setup_scattering_functions<'bsdf, 'arena: 'bsdf>(
//         &self,
//         intersection: &mut Intersection,
//         arena: &'arena DynamicArena<'_, NonSend>,
//     ) -> &'bsdf dyn BSDF<S> {
//         let p = intersection.point;
//         let in_x = p.x.modulo(self.scale.x * 2.0) < self.scale.x;
//         let in_y = p.y.modulo(self.scale.y * 2.0) < self.scale.y;
//         let in_z = p.z.modulo(self.scale.z * 2.0) < self.scale.z;
//         let inside = (in_x == in_y) == in_z;
//         if inside {
//             self.mat1.setup_scattering_functions(intersection, arena)
//         } else {
//             self.mat2.setup_scattering_functions(intersection, arena)
//         }
//     }
// }
