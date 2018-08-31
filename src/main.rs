extern crate vek;
extern crate image;
extern crate rand;
extern crate rayon;

use std::sync::{Arc, RwLock};
use std::f32::consts::PI;

use rand::prelude::*;
use rand::distributions::Uniform;
use rayon::prelude::*;

mod camera;
mod color;
mod hitable;
mod ray;
mod sphere;

use camera::Camera;
use color::Color;
use hitable::{Hitable, HitableList};
use sphere::Sphere;
use ray::Ray;

pub trait RandomInit {
    fn rand(rng: &mut ThreadRng) -> Self;
}

pub type Vec3 = vek::vec::repr_c::Vec3<f32>;
pub type Vec2 = vek::vec::repr_c::Vec2<f32>;

impl RandomInit for Vec3 {
    fn rand(rng: &mut ThreadRng) -> Self {
        let theta = rng.gen_range::<f32>(0.0, 2.0 * PI);
        let phi = rng.gen_range::<f32>(-1.0, 1.0);
        let ophisq = (1.0 - phi * phi).sqrt();
        Vec3::new(ophisq * theta.cos(), ophisq * theta.sin(), phi)
    }
}

const DIMS: (u32, u32) = (1920, 1080);
const SAMPLES: usize = 64;
const MAX_BOUNCES: usize = 5;

fn compute_color(ray: &Ray, hitables: Arc<RwLock<HitableList>>, rng: &mut ThreadRng, bounces: usize) -> Color {
    let ht = &hitables.read().unwrap();
    if bounces < MAX_BOUNCES {
        if let Some(record) = ht.hit(ray, 0.0001..100.0) {
            let bounce = record.n + Vec3::rand(rng);
            compute_color(&Ray::new(record.p, bounce), hitables.clone(), rng, bounces + 1) * 0.5
        } else {
            let mut dir = ray.dir().clone();
            dir.normalize();
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

    let mut world = HitableList::new();
    world.push(Box::new(Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0)));
    world.push(Box::new(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5)));

    let world = Arc::new(RwLock::new(world));

    let mut pixels = vec![Color::zero(); DIMS.0 as usize * DIMS.1 as usize];

    pixels
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, p)| {
            let x = i % DIMS.0 as usize;
            let y = (i - x) / DIMS.0 as usize;
            let col = (0..SAMPLES)
                .into_iter()
                .map(|_| {
                    let mut rng = thread_rng();
                    let uniform = Uniform::new(0.0, 1.0);
                    let (r1, r2) = (uniform.sample(&mut rng), uniform.sample(&mut rng));
                    let uv = Vec3::new((x as f32 + r1) / DIMS.0 as f32, (y as f32 + r2) / DIMS.1 as f32, 0.0);
                    // let uv = Vec3::new(x as f32 / DIMS.0 as f32, y as f32 / DIMS.1 as f32, 0.0);
                    let ray = camera.clone().get_ray(uv);
                    compute_color(&ray, world.clone(), &mut rng, 0)
                })
                .fold(Color::zero(), |a, b| a + b);
            let col = col / SAMPLES as f32;
            *p = col;
        });

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let idx = x + (DIMS.1 - 1 - y) * DIMS.0;
        *pixel = pixels[idx as usize].gamma_correct(2.2).into();
    }

    img.save("render.png").unwrap();
}
