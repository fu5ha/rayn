use rand::prelude::*;

use crate::spectrum::Spectrum;
use crate::hitable::Intersection;
use crate::math::{f0_from_ior, f_schlick, f_schlick_c, saturate, RandomInit, Vec3};
use crate::ray::Ray;

pub trait Material: Send + Sync {
    fn scatter(&self, ray: Ray, intersection: Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent>;
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
    fn scatter(&self, ray: Ray, intersection: Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let norm = intersection.normal;
        let cos = norm.dot(*ray.dir() * -1.0).abs();
        let fresnel = f_schlick(cos, 0.04);
        let (attenuation, specular, bounce) = if rng.gen::<f32>() > fresnel {
            (self.albedo, false, norm + Vec3::rand(rng))
        } else {
            let bounce = ray.dir().reflected(norm) + (Vec3::rand(rng) * self.roughness);
            (Spectrum::one() / bounce.dot(norm).abs(), true, bounce)
        };
        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f: attenuation,
            pdf: 1.0,
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
    fn scatter(&self, ray: Ray, intersection: Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let reflected = ray.dir().reflected(intersection.normal);
        let bounce = reflected + Vec3::rand(rng) * self.roughness;
        let cos = bounce.dot(intersection.normal).abs();
        let attenuation = f_schlick_c(cos, self.f0) / cos;
        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f: attenuation,
            pdf: 1.0,
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
    fn scatter(&self, ray: Ray, intersection: Intersection, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let norm = intersection.normal;
        let (refract_norm, eta, cos) = if ray.dir().dot(norm) > 0.0 {
            (
                norm * -1.0,
                self.ior,
                // 1.0 / intersection.eta,
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
        // println!("{}", fresnel);
        let (attenuation, bounce, specular) = if rng.gen::<f32>() > fresnel {
            let refract = ray.dir().refracted(refract_norm, eta)
                + (Vec3::rand(rng) * self.roughness);
            if refract == Vec3::zero() {
                return None;
            }
            (self.refract_color, refract, false)
        } else {
            let bounce = ray.dir().reflected(norm) + (Vec3::rand(rng) * self.roughness);
            (Spectrum::one() / bounce.dot(norm).abs(), bounce, true)
        };

        Some(ScatteringEvent {
            wi: bounce.normalized(),
            f: attenuation,
            pdf: 1.0,
            specular
        })
    }
}
