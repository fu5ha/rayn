use rand::prelude::*;

use crate::color::Color;
use crate::math::{f0_from_ior, f_schlick, f_schlick_c, saturate, RandomInit, Vec3};
use crate::ray::Ray;

pub trait Material: Send + Sync {
    fn scatter(&self, ray: &Ray, norm: &Vec3) -> Option<(Color, Vec3)>;
}

pub struct Diffuse {
    albedo: Color,
    roughness: f32,
}

impl Diffuse {
    pub fn new(albedo: Color, roughness: f32) -> Self {
        Diffuse { albedo, roughness }
    }
}

impl Material for Diffuse {
    fn scatter(&self, ray: &Ray, norm: &Vec3) -> Option<(Color, Vec3)> {
        let norm = *norm;
        let cos = saturate(norm.normalized().dot(ray.dir().clone().normalized() * -1.0));
        // println!("{}", cos);
        let fresnel = f_schlick(cos, 0.04);
        let bounce = if thread_rng().gen::<f32>() > fresnel {
            norm + Vec3::rand(&mut thread_rng())
        } else {
            ray.dir().reflected(norm) + (Vec3::rand(&mut thread_rng()) * self.roughness)
        };
        Some((self.albedo, bounce))
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
    fn scatter(&self, ray: &Ray, norm: &Vec3) -> Option<(Color, Vec3)> {
        let bounce =
            ray.dir().reflected(norm.clone()) + Vec3::rand(&mut thread_rng()) * self.roughness;
        let cos = saturate(norm.normalized().dot(ray.dir().clone().normalized() * -1.0));
        let attenuation = f_schlick_c(cos, self.f0);
        // let attenuation = self.f0;
        Some((attenuation, bounce))
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
    fn scatter(&self, ray: &Ray, norm: &Vec3) -> Option<(Color, Vec3)> {
        let norm = *norm;
        let (refract_norm, eta, cos) = if ray.dir().dot(norm) > 0.0 {
            (
                norm * -1.0,
                self.ior,
                norm.normalized().dot(ray.dir().clone().normalized()),
            )
        } else {
            (
                norm,
                1.0 / self.ior,
                -norm.normalized().dot(ray.dir().clone().normalized()),
            )
        };
        let f0 = f0_from_ior(self.ior);
        let fresnel = f_schlick(saturate(cos), f0);
        // println!("{}", fresnel);
        let bounce = if thread_rng().gen::<f32>() > fresnel {
            let mut refract = ray.dir().refracted(refract_norm, eta)
                + (Vec3::rand(&mut thread_rng()) * self.roughness);
            if refract == Vec3::zero() {
                refract =
                    ray.dir().reflected(norm) + (Vec3::rand(&mut thread_rng()) * self.roughness);
            }
            refract
        } else {
            ray.dir().reflected(norm) + (Vec3::rand(&mut thread_rng()) * self.roughness)
        };
        Some((self.f0, bounce))
    }
}
