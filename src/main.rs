extern crate image;
extern crate rand;
extern crate rayon;
extern crate vek;

use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

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

use camera::Camera;
use color::Color;
use hitable::{Hitable, HitableList};
use material::{Diffuse, Material, Metal, Refractive};
use math::Vec3;
use ray::Ray;
use sphere::Sphere;

const DIMS: (u32, u32) = (1920, 1080);
const SAMPLES: usize = 2048;
const MAX_BOUNCES: usize = 50;

fn compute_color(ray: &Ray, hitables: Arc<RwLock<HitableList>>, bounces: usize) -> Color {
    let ht = &hitables.read().unwrap();
    if bounces < MAX_BOUNCES {
        if let Some(record) = ht.hit(ray, 0.001..1000.0) {
            let scatter = record.material.scatter(ray, &record.n);
            if let Some((attenuation, bounce)) = scatter {
                compute_color(&Ray::new(record.p, bounce), hitables.clone(), bounces + 1)
                    * attenuation
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

    let pink_diffuse: Arc<Material> = Arc::new(Diffuse::new(Color::new(0.7, 0.3, 0.4), 0.0));
    let ground: Arc<Material> = Arc::new(Diffuse::new(Color::new(0.35, 0.3, 0.45), 0.2));
    let gold: Arc<Material> = Arc::new(Metal::new(Color::new(1.0, 0.9, 0.5), 0.0));
    let gold_rough: Arc<Material> = Arc::new(Metal::new(Color::new(1.0, 0.9, 0.5), 0.2));
    let silver: Arc<Material> = Arc::new(Metal::new(Color::new(0.9, 0.9, 0.9), 0.05));
    let glass: Arc<Material> = Arc::new(Refractive::new(Color::new(0.9, 0.9, 0.9), 0.0, 1.5));
    let glass_rough: Arc<Material> = Arc::new(Refractive::new(Color::new(0.9, 0.9, 0.9), 0.2, 1.5));

    let mut world = HitableList::new();
    world.push(Box::new(Sphere::new(
        Vec3::new(0.0, -200.5, -1.0),
        200.0,
        Arc::clone(&ground),
    )));
    world.push(Box::new(Sphere::new(
        Vec3::new(0.0, 0.0, -1.0),
        0.5,
        Arc::clone(&silver),
    )));
    world.push(Box::new(Sphere::new(
        Vec3::new(-1.0, 0.0, -1.0),
        0.5,
        Arc::clone(&pink_diffuse),
    )));
    world.push(Box::new(Sphere::new(
        Vec3::new(1.0, -0.25, -1.0),
        0.25,
        Arc::clone(&gold),
    )));
    world.push(Box::new(Sphere::new(
        Vec3::new(0.4, -0.375, -0.5),
        0.125,
        Arc::clone(&glass),
    )));
    world.push(Box::new(Sphere::new(
        Vec3::new(0.2, -0.4, -0.35),
        0.1,
        Arc::clone(&glass),
    )));
    world.push(Box::new(Sphere::new(
        Vec3::new(-0.25, -0.375, -0.15),
        0.125,
        Arc::clone(&glass_rough),
    )));
    world.push(Box::new(Sphere::new(
        Vec3::new(-0.5, -0.375, -0.5),
        0.125,
        Arc::clone(&gold_rough),
    )));

    let world = Arc::new(RwLock::new(world));

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
                compute_color(&ray, world.clone(), 0)
            }).fold(Color::zero(), |a, b| a + b);
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

    println!("Done in {} seconds.", time_secs as f32 + time_millis as f32 / 1000.0);

    img.save("render_working.png").unwrap();
}
