use rand::prelude::*;

use crate::spectrum::Spectrum;
use crate::hitable::Intersection;
use crate::math::{f0_from_ior, f_schlick, f_schlick_c, saturate, OrthonormalBasis, RandomSample, Vec3};
use crate::ray::Ray;

pub trait Material: Send + Sync {
    fn scatter(&self, ray: Ray, intersection: &mut Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent>;
    fn le(&self, _wo: Vec3) -> Spectrum { Spectrum::zero() }
    fn internal_ior(&self) -> f32 { 1.0 }
}

pub struct ScatteringEvent {
    pub wi: Vec3,
    pub f: Spectrum,
    pub pdf: f32,
    pub specular: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct MaterialHandle(usize);

pub struct MaterialStore(Vec<Box<dyn Material>>);

impl MaterialStore {
    pub fn new() -> Self {
        MaterialStore(Vec::new())
    }

    pub fn add_material(&mut self, material: Box<dyn Material>) -> MaterialHandle {
        self.0.push(material);
        MaterialHandle(self.0.len() - 1)
    }

    pub fn get(&self, handle: MaterialHandle) -> &dyn Material {
        self.0.get(handle.0).map(|b| b.as_ref()).unwrap()
    }
}

pub struct Dielectric {
    albedo: Spectrum,
    roughness: f32,
}

impl Dielectric {
    pub fn new(albedo: Spectrum, roughness: f32) -> Self {
        Dielectric { albedo, roughness }
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: Ray, intersection: &mut Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let norm = intersection.normal;
        let cos = norm.dot(*ray.dir() * -1.0).abs();
        let fresnel = f_schlick(cos, 0.04);
        let (f, pdf, specular, bounce) = if rng.gen::<f32>() > fresnel {
            // Importance sample with cosine weighted distribution
            let sample = Vec3::cosine_weighted_in_hemisphere(rng, 1.0);
            let bounce = intersection.basis() * sample;
            let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
            let f = self.albedo / std::f32::consts::PI;
            (f, pdf, false, bounce)
        } else {
            let reflection = ray.dir().reflected(norm);
            let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);
            let basis = reflection.get_orthonormal_basis();
            let bounce = basis * sample;
            let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
            let f = Spectrum::one() / bounce.dot(norm).abs() / std::f32::consts::PI;
            (f, pdf, true, bounce)
        };
        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f,
            pdf,
            specular,
        })
    }
}

pub struct Metal {
    f0: Spectrum,
    roughness: f32,
}

impl Metal {
    pub fn new(f0: Spectrum, roughness: f32) -> Self {
        Metal { f0, roughness }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: Ray, intersection: &mut Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);
        let reflection = ray.dir().reflected(intersection.normal);
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
}

pub struct Refractive {
    refract_color: Spectrum,
    ior: f32,
    roughness: f32,
}

impl Refractive {
    pub fn new(refract_color: Spectrum, roughness: f32, ior: f32) -> Self {
        Refractive { refract_color, roughness, ior }
    }
}

impl Material for Refractive {
    fn scatter(&self, ray: Ray, intersection: &mut Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let norm = intersection.normal;
        let (refract_norm, eta, cos) = if ray.dir().dot(norm) > 0.0 {
            (
                norm * -1.0,
                self.ior,
                norm.dot(*ray.dir()),
            )
        } else {
            (
                norm,
                1.0 / self.ior,
                -norm.dot(*ray.dir()),
            )
        };
        let f0 = f0_from_ior(self.ior);
        let fresnel = f_schlick(saturate(cos), f0);
        let sample = Vec3::cosine_weighted_in_hemisphere(rng, self.roughness);
        let (f, pdf, bounce) = if rng.gen::<f32>() > fresnel {
            let refraction = ray.dir().refracted(refract_norm, eta);
            if refraction != Vec3::zero() {
                let basis = refraction.get_orthonormal_basis();
                let bounce = basis * sample;
                let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
                let f = self.refract_color / bounce.dot(norm).abs() / std::f32::consts::PI;
                (f, pdf, bounce)
            } else {
                // Total internal reflection
                reflect_part(ray, sample, norm)
            }
        } else {
            reflect_part(ray, sample, norm)
        };

        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f,
            pdf,
            specular: true,
        })
    }
}

fn reflect_part(ray: Ray, sample: Vec3, norm: Vec3) -> (Spectrum, f32, Vec3) {
    let reflection = ray.dir().reflected(norm);
    let basis = reflection.get_orthonormal_basis();
    let bounce = basis * sample;
    let pdf = sample.dot(Vec3::unit_z()) / std::f32::consts::PI;
    let f = Spectrum::one() / bounce.dot(norm).abs() / std::f32::consts::PI;
    (f, pdf, bounce)
}
