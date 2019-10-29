use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;
use rand::rngs::SmallRng;
use rand::Rng;

use crate::hitable::{HitStore, Intersection, WIntersection};
use crate::material::{Material, MaterialHandle, BSDF};
use crate::math::{Vec2u, Vec3};
use crate::ray::{Ray, WRay};
use crate::spectrum::Srgb;
use crate::world::World;

use wide::f32x4;

pub trait Integrator: Send + Sync {
    fn integrate(
        &self,
        world: &World,
        rng: &mut SmallRng,
        depth: usize,
        material: MaterialHandle,
        intersection: WIntersection,
        bump: &Bump,
        spawned_rays: &mut BumpVec<Ray>,
        output_samples: &mut BumpVec<(Vec2u, Srgb, f32)>,
    );
}

#[derive(Clone, Copy)]
pub struct PathTracingIntegrator {
    pub max_bounces: usize,
}

impl Integrator for PathTracingIntegrator {
    fn integrate(
        &self,
        world: &World,
        rng: &mut SmallRng,
        depth: usize,
        material: MaterialHandle,
        mut intersection: WIntersection,
        bump: &Bump,
        spawned_rays: &mut BumpVec<Ray>,
        output_samples: &mut BumpVec<(Vec2u, Srgb, f32)>,
    ) {
        if depth == 0 {
            intersection.ray.alpha = f32x4::from(1.0);
        }

        let wi = intersection.ray.dir;
        let material = world.materials.get(material);

        let bsdf = material.get_bsdf_at(&mut intersection, bump);

        intersection.ray.radiance += bsdf.le(-wi, &intersection) * intersection.ray.throughput;

        let scattering_event = bsdf.scatter(wi, &intersection, rng);

        let ndl = scattering_event.wi.dot(intersection.normal).abs();

        let mut new_throughput =
            intersection.ray.throughput * scattering_event.f / scattering_event.pdf * ndl;

        let roulette_factor = if depth > 2 {
            let roulette_factor = (f32x4::from(1.0) - intersection.ray.throughput.max_channel())
                .max(f32x4::from(0.05));

            new_throughput /= f32x4::from(1.0) - roulette_factor;

            roulette_factor
        } else {
            f32x4::from(0.0)
        };

        let mut new_rays: [Ray; 4] = intersection.create_rays(scattering_event.wi).into();
        let throughputs: [Srgb; 4] = new_throughput.into();

        for ((ray, new_throughput), roulette_factor) in new_rays
            .iter_mut()
            .zip(throughputs.into_iter())
            .zip(roulette_factor.as_ref().iter())
        {
            if depth >= self.max_bounces || rng.gen::<f32>() < *roulette_factor {
                output_samples.push((ray.tile_coord, ray.radiance, ray.alpha));
                break;
            }

            if !new_throughput.is_nan() {
                ray.throughput = *new_throughput;
            }

            spawned_rays.push(*ray);
        }
    }
}
