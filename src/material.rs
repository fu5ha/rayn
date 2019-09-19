use rand::prelude::*;

use crate::color::Color;
use crate::hitable::HitRecord;
use crate::math::{f0_from_ior, f_schlick, f_schlick_c, saturate, RandomInit, Vec3};
use crate::ray::Ray;

pub struct ScatteringEvent {
    pub out_dir: Vec3,
    pub out_ior: f32,
    pub attenuation: Color,
    pub emission: Color,
}

pub trait Material: Send + Sync {
    fn scatter(&self, ray: &Ray, record: &HitRecord, rng: &mut ThreadRng) -> Option<ScatteringEvent>;
}

pub struct Dielectric {
    albedo: Color,
    roughness: f32,
}

impl Dielectric {
    pub fn new(albedo: Color, roughness: f32) -> Self {
        Dielectric { albedo, roughness }
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, record: &HitRecord, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let norm = record.normal;
        let cos = saturate(norm.dot(*ray.dir() * -1.0));
        let fresnel = f_schlick(cos, 0.04);
        let (attenuation, bounce) = if rng.gen::<f32>() > fresnel {
            (self.albedo, norm + Vec3::rand(rng))
        } else {
            (Color::new(1.0, 1.0, 1.0), ray.dir().reflected(norm) + (Vec3::rand(rng) * self.roughness))
        };
        Some(ScatteringEvent {
            out_dir: bounce.normalized(),
            out_ior: ray.medium_ior(),
            attenuation,
            emission: Color::zero()
        })
    }
}

pub struct Metal {
    f0: Color,
    roughness: f32,
}

impl Metal {
    pub fn new(f0: Color, roughness: f32) -> Self {
        Metal { f0, roughness }
    }
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, record: &HitRecord, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let bounce = ray.dir().reflected(record.normal) + Vec3::rand(rng) * self.roughness;
        let cos = saturate(record.normal.dot(ray.dir() * -1.0));
        let attenuation = f_schlick_c(cos, self.f0);
        Some(ScatteringEvent {
            out_dir: bounce.normalized(),
            out_ior: ray.medium_ior(),
            attenuation,
            emission: Color::zero()
        })
    }
}

pub struct Refractive {
    f0: Color,
    ior: f32,
    roughness: f32,
}

impl Refractive {
    pub fn new(f0: Color, roughness: f32, ior: f32) -> Self {
        Refractive { f0, roughness, ior }
    }
}

impl Material for Refractive {
    fn scatter(&self, ray: &Ray, record: &HitRecord, rng: &mut ThreadRng) -> Option<ScatteringEvent> {
        let norm = record.normal;
        let (refract_norm, eta, cos) = if ray.dir().dot(norm) > 0.0 {
            (
                norm * -1.0,
                self.ior / ray.medium_ior(),
                norm.dot(*ray.dir()),
            )
        } else {
            (
                norm,
                ray.medium_ior() / self.ior,
                -norm.dot(*ray.dir()),
            )
        };
        let f0 = f0_from_ior(self.ior);
        let fresnel = f_schlick(saturate(cos), f0);
        // println!("{}", fresnel);
        let (bounce, out_ior) = if rng.gen::<f32>() > fresnel {
            let refract = ray.dir().refracted(refract_norm, eta)
                + (Vec3::rand(rng) * self.roughness);
            if refract == Vec3::zero() {
                return None;
            }
            (refract, self.ior)
        } else {
            (ray.dir().reflected(norm) + (Vec3::rand(rng) * self.roughness), ray.medium_ior())
        };

        Some(ScatteringEvent {
            out_dir: bounce.normalized(),
            out_ior,
            attenuation: self.f0,
            emission: Color::zero()
        })
    }
}
