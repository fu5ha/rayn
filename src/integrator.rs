use dynamic_arena::{ DynamicArena, NonSend };
use rand::prelude::*;

use crate::spectrum::{ IsSpectrum, Xyz, Rgb };
use crate::world::World;
use crate::ray::Ray;
use crate::hitable::Hitable;
use crate::math::{ Vec3 };

pub trait Integrator: Send + Sync {
    fn integrate<S: IsSpectrum>(
        &self,
        world: &World<S>,
        ray: Ray,
        time: f32,
        col_spect: &mut S,
        col_a: &mut f32,
        back_spect: &mut S,
        normals: &mut Vec3,
        rng: &mut ThreadRng,
        arena: &DynamicArena<'_, NonSend>
    );
}

#[derive(Clone, Copy)]
pub struct PathTracingIntegrator {
    pub max_bounces: usize,
}

impl Integrator for PathTracingIntegrator {
    fn integrate<S: IsSpectrum>(
        &self,
        world: &World<S>,
        mut ray: Ray,
        time: f32,
        luminance: &mut S,
        col_a: &mut f32,
        back_luminance: &mut S,
        normals: &mut Vec3,
        rng: &mut ThreadRng,
        arena: &DynamicArena<'_, NonSend>
    ) {
        let mut throughput = S::one();
        for bounce in 0.. {
            if let Some(mut intersection) = world.hitables.hit(&ray, time, 0.001..1000.0) {
                let wi = *ray.dir();
                let material = world.materials.get(intersection.material);

                material.setup_scattering_functions(&mut intersection, &arena);
                let bsdf = unsafe { intersection.bsdf.assume_init() };

                if bounce == 0 {
                    *normals += intersection.normal;
                }

                *luminance += bsdf.le(-wi, &mut intersection) * throughput;
                *col_a += 1.0;

                let scattering_event = bsdf.scatter(wi, &mut intersection, rng);

                if let Some(se) = scattering_event {
                    if se.pdf == 0.0 || se.f.is_black() {
                        break;
                    }
                    throughput *= se.f / se.pdf * se.wi.dot(intersection.normal).abs();
                    ray = intersection.create_ray(se.wi);
                } else {
                    break;
                }
            } else {
                let dir = ray.dir();
                let t = 0.5 * (dir.y + 1.0);

                let l = throughput * S::from(Xyz::from(Rgb::from(Vec3::lerp(Vec3::one(), Vec3::new(0.5, 0.7, 1.0), t))));
                if bounce == 0 {
                    *back_luminance += l;
                } else {
                    *luminance += l;
                }
                break;
            }

            if bounce >= self.max_bounces {
                break;
            }

            if bounce > 2 {
                let roulette_factor = (1.0 - throughput.max_channel()).max(0.05);
                if rng.gen::<f32>() < roulette_factor {
                    break;
                }
                throughput /= 1.0 - roulette_factor;
            }
        }
    }
}