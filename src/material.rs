use dynamic_arena::{DynamicArena, NonSend};
use rand::prelude::*;

use crate::hitable::Intersection;
use crate::math::{
    f0_from_ior, f_schlick, f_schlick_c, saturate, OrthonormalBasis, RandomSample3d, Vec3,
};
use crate::spectrum::IsSpectrum;

pub trait BSDF<S: IsSpectrum>: Send + Sync {
    fn scatter(
        &self,
        wo: Vec3,
        intersection: &mut Intersection<S>,
        rng: &mut ThreadRng,
    ) -> Option<ScatteringEvent<S>>;
    fn bsdf_pdf(
        &self,
        wo: Vec3,
        wi: Vec3,
        intersection: &mut Intersection<S>,
        rng: &mut ThreadRng,
    ) -> (S, f32);
    fn le(&self, _wo: Vec3, _intersection: &mut Intersection<S>) -> S {
        S::zero()
    }
}

pub trait Material<S: IsSpectrum>: Send + Sync {
    fn setup_scattering_functions<'i, 'a: 'i>(
        &self,
        intersection: &mut Intersection<'i, S>,
        arena: &'a DynamicArena<'_, NonSend>,
    );
}

macro_rules! impl_mat_copy {
    ($t:ident) => {
        impl<S: IsSpectrum> Material<S> for $t<S> {
            fn setup_scattering_functions<'i, 'a: 'i>(
                &self,
                intersection: &mut Intersection<'i, S>,
                arena: &'a DynamicArena<'_, NonSend>,
            ) {
                let bsdf = arena.alloc_copy(*self);
                let bsdf_ptr = intersection.bsdf.as_mut_ptr();
                unsafe { *bsdf_ptr = bsdf };
            }
        }
    };
}

pub struct ScatteringEvent<S> {
    pub wi: Vec3,
    pub f: S,
    pub pdf: f32,
    pub specular: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialHandle(usize);

pub struct MaterialStore<S>(Vec<Box<dyn Material<S>>>);

impl<S: IsSpectrum> MaterialStore<S> {
    pub fn new() -> Self {
        MaterialStore(Vec::new())
    }

    pub fn add_material(&mut self, material: Box<dyn Material<S>>) -> MaterialHandle {
        self.0.push(material);
        MaterialHandle(self.0.len() - 1)
    }

    pub fn get(&self, handle: MaterialHandle) -> &dyn Material<S> {
        self.0.get(handle.0).map(|b| b.as_ref()).unwrap()
    }
}

#[derive(Clone, Copy)]
pub struct Dielectric<S> {
    albedo: S,
    roughness: f32,
}

impl<S> Dielectric<S> {
    pub fn new(albedo: S, roughness: f32) -> Self {
        Dielectric { albedo, roughness }
    }
}

impl<S: IsSpectrum> BSDF<S> for Dielectric<S> {
    fn scatter<'a>(
        &self,
        wo: Vec3,
        intersection: &mut Intersection<'a, S>,
        rng: &mut ThreadRng,
    ) -> Option<ScatteringEvent<S>> {
        let norm = intersection.normal;
        let cos = norm.dot(-wo).abs();
        let fresnel = f_schlick(cos, 0.04);
        let (f, pdf, specular, bounce) = if rng.gen::<f32>() > fresnel {
            // Importance sample with cosine weighted distribution
            let sample = Vec3::cosine_weighted_in_hemisphere(rng, 1.0);
            let bounce = intersection.basis() * sample;
            let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
            let f = self.albedo / std::f32::consts::PI;
            (f, pdf, false, bounce)
        } else {
            let reflection = wo.reflected(norm);
            let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);
            let basis = reflection.get_orthonormal_basis();
            let bounce = basis * sample;
            let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
            let f = S::one() / bounce.dot(norm).abs() / std::f32::consts::PI;
            (f, pdf, true, bounce)
        };
        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f,
            pdf,
            specular,
        })
    }

    fn bsdf_pdf(
        &self,
        wo: Vec3,
        wi: Vec3,
        intersection: &mut Intersection<S>,
        rng: &mut ThreadRng,
    ) -> (S, f32) {
        let norm = intersection.normal;
        let cos = norm.dot(-wo).abs();
        let fresnel = f_schlick(cos, 0.04);
        if rng.gen::<f32>() > fresnel {
            let pdf = norm.dot(wi) / std::f32::consts::PI;
            let f = self.albedo / std::f32::consts::PI;
            (f, pdf)
        } else {
            let reflection = wo.reflected(norm);
            let pdf = reflection.dot(wi).abs() / std::f32::consts::PI;
            let f = S::one() / wi.dot(norm).abs() / std::f32::consts::PI;
            (f, pdf)
        }
    }
}

impl_mat_copy!(Dielectric);

#[derive(Clone, Copy)]
pub struct Metal<S> {
    f0: S,
    roughness: f32,
}

impl<S> Metal<S> {
    pub fn new(f0: S, roughness: f32) -> Self {
        Metal { f0, roughness }
    }
}

impl<S: IsSpectrum> BSDF<S> for Metal<S> {
    fn scatter<'a>(
        &self,
        wo: Vec3,
        intersection: &mut Intersection<'a, S>,
        rng: &mut ThreadRng,
    ) -> Option<ScatteringEvent<S>> {
        let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);
        let reflection = wo.reflected(intersection.normal);
        let basis = reflection.get_orthonormal_basis();
        let bounce = basis * sample;
        let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
        let cos = bounce.dot(intersection.normal).abs();
        let f = f_schlick_c(cos, self.f0) / cos / std::f32::consts::PI;
        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f,
            pdf,
            specular: true,
        })
    }

    fn bsdf_pdf(
        &self,
        wo: Vec3,
        wi: Vec3,
        intersection: &mut Intersection<S>,
        _rng: &mut ThreadRng,
    ) -> (S, f32) {
        let reflection = wo.reflected(intersection.normal);
        let pdf = wi.dot(reflection).abs() / std::f32::consts::PI;
        let cos = wi.dot(intersection.normal).abs();
        let f = f_schlick_c(cos, self.f0) / cos / std::f32::consts::PI;
        (f, pdf)
    }
}

impl_mat_copy!(Metal);

#[derive(Clone, Copy)]
pub struct Refractive<S> {
    refract_color: S,
    ior: f32,
    roughness: f32,
}

impl<S> Refractive<S> {
    pub fn new(refract_color: S, roughness: f32, ior: f32) -> Self {
        Refractive {
            refract_color,
            roughness,
            ior,
        }
    }
}

impl_mat_copy!(Refractive);

impl<S: IsSpectrum> BSDF<S> for Refractive<S> {
    fn scatter<'a>(
        &self,
        wo: Vec3,
        intersection: &mut Intersection<'a, S>,
        rng: &mut ThreadRng,
    ) -> Option<ScatteringEvent<S>> {
        let norm = intersection.normal;
        let odn = wo.dot(norm);
        let (refract_norm, eta, cos) = if odn > 0.0 {
            (norm * -1.0, self.ior, odn)
        } else {
            (norm, 1.0 / self.ior, -odn)
        };
        let f0 = f0_from_ior(self.ior);
        let fresnel = f_schlick(saturate(cos), f0);
        let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);
        let (f, pdf, bounce) = if rng.gen::<f32>() > fresnel {
            let refraction = wo.refracted(refract_norm, eta);
            if refraction != Vec3::zero() {
                let basis = refraction.get_orthonormal_basis();
                let bounce = basis * sample;
                let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
                let f = self.refract_color / bounce.dot(norm).abs() / std::f32::consts::PI;
                (f, pdf, bounce)
            } else {
                // Total internal reflection
                reflect_part(wo, sample, norm)
            }
        } else {
            reflect_part(wo, sample, norm)
        };

        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f,
            pdf,
            specular: true,
        })
    }

    fn bsdf_pdf(
        &self,
        wo: Vec3,
        wi: Vec3,
        intersection: &mut Intersection<S>,
        rng: &mut ThreadRng,
    ) -> (S, f32) {
        let norm = intersection.normal;
        let (refract_norm, eta, cos) = if wo.dot(norm) > 0.0 {
            (norm * -1.0, self.ior, norm.dot(wo))
        } else {
            (norm, 1.0 / self.ior, -norm.dot(wo))
        };
        let f0 = f0_from_ior(self.ior);
        let fresnel = f_schlick(saturate(cos), f0);
        let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);
        if rng.gen::<f32>() > fresnel {
            let refraction = wo.refracted(refract_norm, eta);
            if refraction != Vec3::zero() {
                let basis = refraction.get_orthonormal_basis();
                let bounce = basis * sample;
                let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
                let f = self.refract_color / bounce.dot(norm).abs() / std::f32::consts::PI;
                (f, pdf)
            } else {
                // Total internal reflection
                reflect_brdf_pdf(wo, wi, norm)
            }
        } else {
            reflect_brdf_pdf(wo, wi, norm)
        }
    }
}

fn reflect_brdf_pdf<S: IsSpectrum>(wo: Vec3, wi: Vec3, norm: Vec3) -> (S, f32) {
    let reflection = wo.reflected(norm);
    let pdf = wi.dot(reflection) / std::f32::consts::PI;
    let f = S::one() / wi.dot(norm).abs() / std::f32::consts::PI;
    (f, pdf)
}

fn reflect_part<S: IsSpectrum>(wo: Vec3, sample: Vec3, norm: Vec3) -> (S, f32, Vec3) {
    let reflection = wo.reflected(norm);
    let basis = reflection.get_orthonormal_basis();
    let bounce = basis * sample;
    let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
    let f = S::one() / bounce.dot(norm).abs() / std::f32::consts::PI;
    (f, pdf, bounce)
}

#[derive(Clone, Copy)]
pub struct Emissive<S, I> {
    pub emission: S,
    pub inner: I,
}

impl<S: IsSpectrum, I: BSDF<S> + Copy + 'static> Material<S> for Emissive<S, I> {
    fn setup_scattering_functions<'i, 'a: 'i>(
        &self,
        intersection: &mut Intersection<'i, S>,
        arena: &'a DynamicArena<'_, NonSend>,
    ) {
        let bsdf = arena.alloc_copy(*self);
        let bsdf_ptr = intersection.bsdf.as_mut_ptr();
        unsafe { *bsdf_ptr = bsdf };
    }
}

impl<S, I> Emissive<S, I> {
    pub fn new(emission: S, inner: I) -> Self {
        Emissive { emission, inner }
    }
}

impl<S, I> BSDF<S> for Emissive<S, I>
where
    S: IsSpectrum,
    I: BSDF<S>,
{
    fn scatter<'a>(
        &self,
        wo: Vec3,
        intersection: &mut Intersection<'a, S>,
        rng: &mut ThreadRng,
    ) -> Option<ScatteringEvent<S>> {
        self.inner.scatter(wo, intersection, rng)
    }

    fn bsdf_pdf(
        &self,
        wo: Vec3,
        wi: Vec3,
        intersection: &mut Intersection<S>,
        rng: &mut ThreadRng,
    ) -> (S, f32) {
        self.inner.bsdf_pdf(wo, wi, intersection, rng)
    }

    fn le(&self, _wo: Vec3, _intersection: &mut Intersection<S>) -> S {
        self.emission
    }
}

#[derive(Copy, Clone)]
pub struct Checkerboard3d<M1, M2> {
    pub mat1: M1,
    pub mat2: M2,
    pub scale: Vec3,
}

impl<M1, M2> Checkerboard3d<M1, M2> {
    pub fn new(scale: Vec3, mat1: M1, mat2: M2) -> Self {
        Checkerboard3d { scale, mat1, mat2 }
    }
}

trait ModuloSigned {
    fn modulo(&self, n: Self) -> Self;
}

impl<T> ModuloSigned for T
where
    T: std::ops::Add<Output = T> + std::ops::Rem<Output = T> + Copy,
{
    fn modulo(&self, n: T) -> T {
        (*self % n + n) % n
    }
}

impl<S, M1, M2> Material<S> for Checkerboard3d<M1, M2>
where
    S: IsSpectrum,
    M1: Material<S>,
    M2: Material<S>,
{
    fn setup_scattering_functions<'i, 'a: 'i>(
        &self,
        intersection: &mut Intersection<'i, S>,
        arena: &'a DynamicArena<'_, NonSend>,
    ) {
        let p = intersection.point;
        let in_x = p.x.modulo(self.scale.x * 2.0) < self.scale.x;
        let in_y = p.y.modulo(self.scale.y * 2.0) < self.scale.y;
        let in_z = p.z.modulo(self.scale.z * 2.0) < self.scale.z;
        let inside = (in_x == in_y) == in_z;
        if inside {
            self.mat1.setup_scattering_functions(intersection, arena);
        } else {
            self.mat2.setup_scattering_functions(intersection, arena);
        }
    }
}
