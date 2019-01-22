use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;

mod camera;
mod color;
mod hitable;
mod material;
mod math;
mod ray;
mod sphere;

use crate::camera::Camera;
use crate::color::Color;
use crate::hitable::{Hitable, HitableList};
use crate::material::{Diffuse, Metal, Refractive};
use crate::math::Vec3;
use crate::ray::Ray;
use crate::sphere::Sphere;

use lazy_static::lazy_static;
lazy_static! {
    static ref pink_diffuse: Diffuse = Diffuse::new(Color::new(0.7, 0.3, 0.4), 0.0);
    static ref ground: Diffuse = Diffuse::new(Color::new(0.35, 0.3, 0.45), 0.2);
    static ref gold: Metal = Metal::new(Color::new(1.0, 0.9, 0.5), 0.0);
    static ref gold_rough: Metal = Metal::new(Color::new(1.0, 0.9, 0.5), 0.2);
    static ref silver: Metal = Metal::new(Color::new(0.9, 0.9, 0.9), 0.05);
    static ref glass: Refractive = Refractive::new(Color::new(0.9, 0.9, 0.9), 0.0, 1.5);
    static ref glass_rough: Refractive = Refractive::new(Color::new(0.9, 0.9, 0.9), 0.2, 1.5);
    static ref WORLD: HitableList = {
        let mut world = HitableList::new();
        world.push(Box::new(Sphere::new(
            Vec3::new(0.0, -200.5, -1.0),
            200.0,
            &*ground,
        )));
        world.push(Box::new(Sphere::new(
            Vec3::new(0.0, 0.0, -1.0),
            0.5,
            &*silver,
        )));
        world.push(Box::new(Sphere::new(
            Vec3::new(-1.0, 0.0, -1.0),
            0.5,
            &*pink_diffuse,
        )));
        world.push(Box::new(Sphere::new(
            Vec3::new(1.0, -0.25, -1.0),
            0.25,
            &*gold,
        )));
        world.push(Box::new(Sphere::new(
            Vec3::new(0.4, -0.375, -0.5),
            0.125,
            &*glass,
        )));
        world.push(Box::new(Sphere::new(
            Vec3::new(0.2, -0.4, -0.35),
            0.1,
            &*glass,
        )));
        world.push(Box::new(Sphere::new(
            Vec3::new(-0.25, -0.375, -0.15),
            0.125,
            &*glass_rough,
        )));
        world.push(Box::new(Sphere::new(
            Vec3::new(-0.5, -0.375, -0.5),
            0.125,
            &*gold_rough,
        )));
        world
    };
}

const DIMS: (u32, u32) = (1920, 1080);
const SAMPLES: usize = 128;
const MAX_BOUNCES: usize = 256;

fn compute_color(ray: &Ray, bounces: usize) -> Color {
    if bounces < MAX_BOUNCES {
        if let Some(record) = WORLD.hit(ray, 0.001..1000.0) {
            let scatter = record.material.scatter(ray, &record.n);
            if let Some((attenuation, bounce)) = scatter {
                compute_color(&Ray::new(record.p, bounce), bounces + 1) * attenuation
            } else {
                Color::zero()
            }
        } else {
            let dir = ray.dir().clone().normalized();
            let t = 0.5 * (dir.y + 1.0);

            Color(Vec3::lerp(Vec3::one(), Vec3::new(0.5, 0.7, 1.0), t))
        }
    } else {
        Color::zero()
    }
}

fn main() {
    let mut img = image::RgbImage::new(DIMS.0, DIMS.1);

    let camera = Arc::new(Camera::new(DIMS.0 as f32 / DIMS.1 as f32));

    let mut pixels = vec![Color::zero(); DIMS.0 as usize * DIMS.1 as usize];

    let mutated = AtomicUsize::new(0);
    let start = Instant::now();

    pixels.par_iter_mut().enumerate().for_each(|(i, p)| {
        let x = i % DIMS.0 as usize;
        let y = (i - x) / DIMS.0 as usize;
        let col = (0..SAMPLES)
            .into_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let uniform = Uniform::new(0.0, 1.0);
                let (r1, r2) = (uniform.sample(&mut rng), uniform.sample(&mut rng));
                let uv = Vec3::new(
                    (x as f32 + r1) / DIMS.0 as f32,
                    (y as f32 + r2) / DIMS.1 as f32,
                    0.0,
                );
                // let uv = Vec3::new(x as f32 / DIMS.0 as f32, y as f32 / DIMS.1 as f32, 0.0);
                let ray = camera.clone().get_ray(uv);
                compute_color(&ray, 0)
            })
            .fold(Color::zero(), |a, b| a + b);
        let col = col / SAMPLES as f32;
        *p = col;
        if i % 100000 == 0 {
            let n = mutated.fetch_add(1, Ordering::Relaxed);
            println!(
                "{}% finished...",
                n as f32 / (DIMS.0 * DIMS.1) as f32 * 100.0 * 100000.0
            );
        }
    });

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let idx = x + (DIMS.1 - 1 - y) * DIMS.0;
        *pixel = pixels[idx as usize].gamma_correct(2.2).into();
    }

    let time = Instant::now() - start;
    let time_secs = time.as_secs();
    let time_millis = time.subsec_millis();

    println!(
        "Done in {} seconds.",
        time_secs as f32 + time_millis as f32 / 1000.0
    );
    let args: Vec<String> = std::env::args().into_iter().collect();
    let default = String::from("render.png");
    let filename = args.get(1).unwrap_or(&default);
    println!("Saving to {}", filename);

    img.save(filename).unwrap();
}
