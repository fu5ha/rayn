use generic_array::typenum::*;

mod animation;
mod camera;
mod film;
mod filter;
mod hitable;
mod integrator;
mod light;
mod material;
mod math;
mod ray;
mod sampler;
mod sdf;
mod spectrum;
mod sphere;
mod world;

use camera::{CameraHandle, CameraStore, OrthographicCamera};
use film::{ChannelKind, Film};
use filter::BlackmanHarrisFilter;
use hitable::HitableStore;
use integrator::PathTracingIntegrator;
use light::{Light, SphereLight};
use material::{Dielectric, MaterialStore, Sky};
use math::{Extent2u, Vec2, Vec3};
use sdf::{BoxFold, MandelBox, SphereFold, TracedSDF};
use spectrum::Srgb;
use sphere::Sphere;
use world::World;

use std::time::Instant;

const RES: (usize, usize) = (1920, 1080);
const SAMPLES: usize = 2;

const MB_ITERS: usize = 20;

fn setup() -> (CameraHandle, World) {
    let mut materials = MaterialStore::new();

    let grey = materials.add_material(Dielectric::new_remap(Srgb::new(0.2, 0.2, 0.2), 0.8));

    let sky = materials.add_material(Sky::new(
        Srgb::new(0.6, 0.3, 0.5) * 5.0,
        Srgb::new(0.3, 0.2, 0.6) * 3.0,
    ));

    let mut hitables = HitableStore::new();

    hitables.push(Sphere::new(Vec3::new(0.0, 0.0, 0.0), 300.0, sky));

    hitables.push(TracedSDF::new(
        MandelBox::new(MB_ITERS, BoxFold::new(1.0), SphereFold::new(0.5, 1.0), -2.0),
        grey,
    ));

    let mut lights: Vec<Box<dyn Light>> = Vec::new();

    let bluesun = Srgb::new(1.5, 3.0, 5.0) * 18000.0;
    let pink = Srgb::new(4.5, 1.5, 3.0) * 4.0;
    let blue = Srgb::new(1.5, 3.5, 4.5) * 4.0;

    lights.push(Box::new(SphereLight::new(
        Vec3::new(2.65, 1.5, -1.0) * 50.0,
        1.0,
        bluesun,
    )));

    // lights.push(Box::new(SphereLight::new(
    //     Vec3::new(-1.0, 2.0, 2.65) * 50.0,
    //     1.0,
    //     sun,
    // )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(1.5, -0.35, 1.95),
        0.1,
        blue,
    )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(1.5, 0.35, 1.95),
        0.1,
        pink,
    )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(1.95, -0.35, 1.5),
        0.1,
        pink,
    )));

    lights.push(Box::new(SphereLight::new(
        Vec3::new(1.95, 0.35, 1.5),
        0.1,
        blue,
    )));

    let res = Vec2::new(RES.0 as f32, RES.1 as f32);
    // 1
    let camera = OrthographicCamera::new(
        res,
        11.0 / 4.0,
        Vec3::new(9.5, -3.5, 9.5),
        Vec3::new(0.0, 0.8, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );

    let mut cameras = CameraStore::new();

    let camera = cameras.add_camera(Box::new(camera));

    (
        camera,
        World {
            materials,
            hitables,
            lights,
            cameras,
        },
    )
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();

    let (camera, world) = setup();

    let mut film = Film::<U4>::new(
        &[
            ChannelKind::Color,
            ChannelKind::Alpha,
            ChannelKind::Background,
            ChannelKind::WorldNormal,
        ],
        Extent2u::new(RES.0, RES.1),
    )
    .unwrap();

    let frame_rate = 24;
    let frame_range = 1..2;
    let shutter_speed = 1.0 / 24.0;

    let filter = BlackmanHarrisFilter::new(1.5);
    // let filter = BoxFilter::default();
    let integrator = PathTracingIntegrator { max_bounces: 5 };

    for frame in frame_range {
        let start = Instant::now();

        let frame_start = frame as f32 * (1.0 / frame_rate as f32);
        let frame_end = frame_start + shutter_speed;

        film.render_frame_into(
            &world,
            camera,
            &integrator,
            &filter,
            Extent2u::new(16, 16),
            frame,
            frame_start..frame_end,
            SAMPLES,
        );

        let time = Instant::now() - start;
        let time_secs = time.as_secs();
        let time_millis = time.subsec_millis();

        println!(
            "Done in {} seconds.",
            time_secs as f32 + time_millis as f32 / 1000.0
        );

        println!("Post processing image...");

        film.save_to(
            &[ChannelKind::WorldNormal, ChannelKind::Color],
            "renders",
            format!("{}_spp", SAMPLES * 4),
            false,
        )
        .unwrap();
    }
}
