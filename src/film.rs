use bumpalo::{collections::Vec as BumpVec, Bump};

use generic_array::{ArrayLength, GenericArray};

use rand::prelude::*;

use crate::camera::CameraHandle;
use crate::filter::{Filter, FilterImportanceSampler};
use crate::hitable::HitStore;
use crate::integrator::Integrator;
use crate::math::{f32x4, Aabru, Extent2u, Vec2, Vec2u, Vec3, Wec2};
use crate::ray::{Ray, WRay};
use crate::spectrum::Srgb;
use crate::world::World;

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

macro_rules! declare_channels {
    {
        $($name:ident => {
            storage: $storage:ident,
            init: $initialize:expr,
        }),+
    } => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum ChannelKind {
            $($name,)+
        }

        #[derive(Debug)]
        pub enum ChannelSample {
            $($name($storage),)+
        }

        pub enum ChannelTileStorage {
            $($name(Vec<($storage)>),)+
        }

        impl ChannelTileStorage {
            fn new(kind: ChannelKind, res: Extent2u) -> Self {
                use ChannelKind::*;
                match kind {
                    $($name => Self::$name(vec![($initialize); res.w * res.h]),)+
                }
            }

            fn add_sample(&mut self, idx: usize, sample: &ChannelSample) {
                match (self, sample) {
                    $((ChannelTileStorage::$name(ref mut buf), ChannelSample::$name(sample)) => {
                        buf[idx] += *sample;
                    },)+
                    _ => (),
                }
            }
        }

        pub enum ChannelStorage {
            $($name(Vec<$storage>),)+
        }

        impl ChannelStorage {
            fn new(kind: ChannelKind, res: Extent2u) -> Self {
                use ChannelKind::*;
                match kind {
                    $($name => Self::$name(vec![$initialize; res.w * res.h]),)+
                }
            }

            pub fn kind(&self) -> ChannelKind {
                match *self {
                    $( ChannelStorage::$name(_) => ChannelKind::$name, )+
                }
            }

            pub fn copy_from_tile(&mut self, other: &ChannelTileStorage, full_res: Extent2u, tile_bounds: Aabru, samples: usize) -> Result<(), ()> {
                let extent = tile_bounds.size();
                match (self, other) {
                    $( (ChannelStorage::$name(this_buf), ChannelTileStorage::$name(tile_buf)) => {
                        for x in 0..extent.w {
                            for y in 0..extent.h {
                                let tile_idx = x + y * extent.w;
                                let this_idx = (tile_bounds.min.x + x) + (tile_bounds.min.y + y) * full_res.w;
                                let tile_samp_sum = tile_buf[tile_idx];
                                this_buf[this_idx] = tile_samp_sum / samples as f32;
                            }
                        }
                        Ok(())
                    }, )+
                    _ => Err(())
                }
            }
        }
    }
}

declare_channels! {
    Color => {
        storage: Srgb,
        init: Srgb::zero(),
    },
    Alpha => {
        storage: f32,
        init: 0f32,
    },
    Background => {
        storage: Srgb,
        init: Srgb::zero(),
    },
    WorldNormal => {
        storage: Vec3,
        init: Vec3::zero(),
    }
}

macro_rules! channel_storage_index {
    ($storage:expr, $channel:ident, $idx:expr) => {
        if let ChannelStorage::$channel(x) = &$storage[$idx] {
            x
        } else {
            panic!("Attempted to index into channel storage array with wrong channel type.");
        }
    };
}

pub struct Tile<N: ArrayLength<ChannelTileStorage>> {
    index: usize,
    epoch: usize,
    channels: GenericArray<ChannelTileStorage, N>,
    raster_bounds: Aabru,
    raster_extent: Extent2u,
    screen_to_ndc_size: Vec2,
}

impl<N: ArrayLength<ChannelTileStorage>> Tile<N> {
    pub fn new<IC>(
        index: usize,
        epoch: usize,
        channels: IC,
        res: Extent2u,
        raster_bounds: Aabru,
    ) -> Self
    where
        IC: std::iter::ExactSizeIterator<Item = ChannelKind>,
    {
        let screen_to_ndc_size = Vec2::new(1.0 / res.w as f32, 1.0 / res.h as f32);

        Tile {
            index,
            epoch,
            channels: GenericArray::from_exact_iter(
                channels.map(|kind| ChannelTileStorage::new(kind, raster_bounds.size())),
            )
            .expect("Incorrect number of channels passed to tile creation"),
            raster_bounds,
            raster_extent: raster_bounds.size(),
            screen_to_ndc_size,
        }
    }

    pub fn add_sample(&mut self, tile_coord: Vec2u, sample: ChannelSample) {
        let idx = tile_coord.x + tile_coord.y * self.raster_extent.w;
        for channel in self.channels.iter_mut() {
            channel.add_sample(idx, &sample);
        }
    }
}

pub struct Film<N: ArrayLength<ChannelStorage>> {
    channel_indices: HashMap<ChannelKind, usize>,
    channels: Mutex<GenericArray<ChannelStorage, N>>,
    progressive_epoch: usize,
    this_epoch_tiles_finished: AtomicUsize,
    res: Extent2u,
}

impl<'a, N: ArrayLength<ChannelStorage>> Film<N> {
    pub fn new(channels: &[ChannelKind], res: Extent2u) -> Result<Self, String> {
        let mut channel_indices = HashMap::new();
        for (i, kind) in channels.iter().enumerate() {
            if channel_indices.insert(*kind, i).is_some() {
                return Err(format!("Attempted to create multiple {:?} channels", *kind));
            }
        }
        Ok(Film {
            channel_indices,
            channels: Mutex::new(
                GenericArray::from_exact_iter(
                    channels.iter().map(|kind| ChannelStorage::new(*kind, res)),
                )
                .expect("Generic type length does not match the number of channels."),
            ),
            progressive_epoch: 0,
            this_epoch_tiles_finished: AtomicUsize::new(0),
            res,
        })
    }

    pub fn save_to<P: AsRef<std::path::Path>, IS: Into<String>>(
        &self,
        write_channels: &[ChannelKind],
        output_folder: P,
        base_name: IS,
        transparent_background: bool,
    ) -> Result<(), String> {
        let base_name = base_name.into();

        let channels = self.channels.lock().unwrap();

        for kind in write_channels.iter() {
            match *kind {
                ChannelKind::Color => {
                    let color_idx = self.channel_indices.get(&ChannelKind::Color);
                    let alpha_idx = self.channel_indices.get(&ChannelKind::Alpha);
                    let bg_idx = self.channel_indices.get(&ChannelKind::Background);

                    match (color_idx, alpha_idx, bg_idx, transparent_background) {
                        (Some(&color_idx), Some(&alpha_idx), _, true) => {
                            let color_buf = &channel_storage_index!(channels, Color, color_idx);
                            let alpha_buf = &channel_storage_index!(channels, Alpha, alpha_idx);
                            let mut img =
                                image::RgbaImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let col = color_buf[idx];
                                let a = alpha_buf[idx];
                                let rgb = (col * a).saturated().gamma_corrected(2.2);
                                let a = a.powf(1.0 / 2.2);
                                *pixel = image::Rgba([
                                    (rgb.x * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.y * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.z * 255.0).min(255.0).max(0.0) as u8,
                                    (a * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder
                                .as_ref()
                                .join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        }
                        (Some(&color_idx), _, Some(&bg_idx), false) => {
                            let color_buf = channel_storage_index!(channels, Color, color_idx);
                            let bg_buf = channel_storage_index!(channels, Background, bg_idx);
                            let mut img =
                                image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let i = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let col = color_buf[i];
                                let bg = bg_buf[i];
                                let rgb = (col + bg).saturated().gamma_corrected(2.2);
                                *pixel = image::Rgb([
                                    (rgb.x * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.y * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.z * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder
                                .as_ref()
                                .join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        }
                        (Some(&color_idx), _, None, false) => {
                            let color_buf = channel_storage_index!(channels, Color, color_idx);
                            let mut img =
                                image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                            for (x, y, pixel) in img.enumerate_pixels_mut() {
                                let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                                let rgb = color_buf[idx].gamma_corrected(2.2);
                                *pixel = image::Rgb([
                                    (rgb.x * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.y * 255.0).min(255.0).max(0.0) as u8,
                                    (rgb.z * 255.0).min(255.0).max(0.0) as u8,
                                ]);
                            }
                            let filename = output_folder
                                .as_ref()
                                .join(format!("{}_color.png", base_name.clone()));
                            println!("Saving to {}...", filename.display());
                            img.save(filename).unwrap();
                        }
                        _ => {
                            return Err(String::from(
                                "Attempted to write Color channel with insufficient channels",
                            ))
                        }
                    }
                }
                ChannelKind::Background => {
                    let idx = *self
                        .channel_indices
                        .get(&ChannelKind::Background)
                        .ok_or_else(|| {
                            String::from(
                                "Attempted to write Background channel but it didn't exist",
                            )
                        })?;
                    let buf = channel_storage_index!(channels, Background, idx);
                    let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let rgb = buf[idx].saturated().gamma_corrected(2.2);
                        *pixel = image::Rgb([
                            (rgb.x * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.y * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.z * 255.0).min(255.0).max(0.0) as u8,
                        ]);
                    }
                    let filename = output_folder
                        .as_ref()
                        .join(format!("{}_background.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                }
                ChannelKind::WorldNormal => {
                    let idx = *self
                        .channel_indices
                        .get(&ChannelKind::WorldNormal)
                        .ok_or_else(|| {
                            String::from(
                                "Attempted to write WorldNormal channel but it didn't exist",
                            )
                        })?;
                    let buf = channel_storage_index!(channels, WorldNormal, idx);
                    let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let vec = buf[idx];
                        let rgb = Srgb::from(vec * 0.5 + Vec3::new(0.5, 0.5, 0.5));
                        *pixel = image::Rgb([
                            (rgb.x * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.y * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.z * 255.0).min(255.0).max(0.0) as u8,
                        ]);
                    }
                    let filename = output_folder
                        .as_ref()
                        .join(format!("{}_normal.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                }
                ChannelKind::Alpha => {
                    let idx = *self
                        .channel_indices
                        .get(&ChannelKind::Alpha)
                        .ok_or_else(|| {
                            String::from("Attempted to write Alpha channel but it didn't exist")
                        })?;
                    let buf = channel_storage_index!(channels, Alpha, idx);
                    let mut img = image::GrayImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                        let a = buf[idx];
                        *pixel = image::Luma([(a * 255.0).min(255.0).max(0.0) as u8]);
                    }
                    let filename = output_folder
                        .as_ref()
                        .join(format!("{}_alpha.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                }
            }
        }
        Ok(())
    }
}

impl<'a, N: ArrayLength<ChannelStorage> + ArrayLength<ChannelTileStorage>> Film<N> {
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame_into<I, F>(
        &'a mut self,
        world: &World,
        camera: CameraHandle,
        integrator: &I,
        filter: &F,
        tile_size: Extent2u,
        time_range: Range<f32>,
        samples: usize,
    ) where
        F: Filter + Copy + Send,
        I: Integrator,
    {
        let camera = world.cameras.get(camera);
        let mut tiles = Vec::new();

        let rem = Vec2u::new((self.res.w) % tile_size.w, (self.res.h) % tile_size.h);
        {
            let mut idx = 0;
            let channels = self.channels.lock().unwrap();
            for tile_x in 0..((self.res.w + rem.x) / tile_size.w) {
                for tile_y in 0..((self.res.h + rem.y) / tile_size.h) {
                    let start = Vec2u::new(tile_x * tile_size.w, tile_y * tile_size.h);
                    let end = Vec2u::new(
                        (start.x + tile_size.w).min(self.res.w),
                        (start.y + tile_size.h).min(self.res.h),
                    );
                    let tile_bounds = Aabru {
                        min: start,
                        max: end,
                    };

                    let tile = Tile::new(
                        idx,
                        self.progressive_epoch,
                        channels.iter().map(|c| c.kind()),
                        self.res,
                        tile_bounds,
                    );
                    tiles.push(tile);

                    idx += 1;
                }
            }
        }

        let fis = FilterImportanceSampler::new(filter);

        self.integrate_tiles(tiles, samples * 4, |tile| {
            let mut rng = SmallRng::seed_from_u64(tile.index as u64);
            // let mut rng = SmallRng::from_rng(thread_rng()).unwrap();

            let ray_bump = Bump::new();
            let mut spawned_rays = BumpVec::new_in(&ray_bump);
            let mut spawned_wrays = BumpVec::new_in(&ray_bump);
            let shading_point_bump = Bump::new();
            let mut wintersections = BumpVec::new_in(&shading_point_bump);
            let sample_bump = Bump::new();
            let mut new_samples = BumpVec::new_in(&sample_bump);
            let hit_bump = Bump::new();
            let mut hit_store = HitStore::from_hitable_store(&hit_bump, &world.hitables);
            let mut bsdf_bump = Bump::new();

            for x in tile.raster_bounds.min.x..tile.raster_bounds.max.x {
                for y in tile.raster_bounds.min.y..tile.raster_bounds.max.y {
                    let tile_coord = Vec2u::new(x, y) - tile.raster_bounds.min;

                    for _ in 0..samples {
                        // let raster_pixel = Vec2u::new(x, y);
                        // sampler.begin_pixel(raster_pixel);
                        let ndcs = Wec2::from([
                            sample_uv(x, y, tile.screen_to_ndc_size, &fis, &mut rng),
                            sample_uv(x, y, tile.screen_to_ndc_size, &fis, &mut rng),
                            sample_uv(x, y, tile.screen_to_ndc_size, &fis, &mut rng),
                            sample_uv(x, y, tile.screen_to_ndc_size, &fis, &mut rng),
                        ]);

                        let times = f32x4::from([
                            rng.gen_range(time_range.start, time_range.end),
                            rng.gen_range(time_range.start, time_range.end),
                            rng.gen_range(time_range.start, time_range.end),
                            rng.gen_range(time_range.start, time_range.end),
                        ]);

                        let rays = camera.get_rays(tile_coord, ndcs, times, &mut rng);

                        spawned_wrays.push(rays);
                    }
                }
            }

            for depth in 0.. {
                bsdf_bump.reset();

                if spawned_wrays.is_empty() {
                    break;
                }

                hit_store.reset();

                for wray in spawned_wrays.drain(..) {
                    world.hitables.add_hits(
                        wray,
                        f32x4::from(0.0001)..f32x4::from(500.0),
                        &mut hit_store,
                    );
                }

                hit_store.process_hits(&world.hitables, &mut wintersections);

                for (mat_id, wshading_point) in wintersections.drain(..) {
                    integrator.integrate(
                        world,
                        &mut rng,
                        depth,
                        mat_id,
                        wshading_point,
                        &bsdf_bump,
                        &mut spawned_rays,
                        &mut new_samples,
                    );
                }

                for (tile_coord, sample) in new_samples.drain(..) {
                    tile.add_sample(tile_coord, sample);
                }

                while spawned_rays.len() % 4 != 0 {
                    spawned_rays.push(Ray::new_invalid());
                }

                for rays in spawned_rays[0..].chunks_exact(4) {
                    // Safe because we just ensured that it has the correct length
                    let wray = WRay::from(unsafe {
                        [
                            *rays.get_unchecked(0),
                            *rays.get_unchecked(1),
                            *rays.get_unchecked(2),
                            *rays.get_unchecked(3),
                        ]
                    });

                    spawned_wrays.push(wray);
                }
                spawned_rays.clear();
            }
        });
    }

    fn integrate_tiles<FN>(&mut self, tiles: Vec<Tile<N>>, samples: usize, integrate_tile: FN)
    where
        FN: FnOnce(&mut Tile<N>) + Send + Sync + Copy,
    {
        let num_tiles = tiles.len();

        {
            let this = &*self;
            rayon::scope_fifo(|scope| {
                for (idx, mut tile) in tiles.into_iter().enumerate() {
                    scope.spawn_fifo(move |_| {
                        integrate_tile(&mut tile);

                        this.tile_finished(tile, num_tiles, idx, samples)
                    })
                }
            });
        }

        while !self.this_epoch_tiles_finished.load(Ordering::Relaxed) == num_tiles {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        self.this_epoch_tiles_finished.store(0, Ordering::Relaxed);
        self.progressive_epoch += 1;
    }

    fn tile_finished(&self, tile: Tile<N>, num_tiles: usize, tile_idx: usize, samples: usize) {
        if self.progressive_epoch != tile.epoch {
            panic!(
                "Epoch mismatch! Expected: {}, got: {}",
                self.progressive_epoch, tile.epoch
            );
        }

        let tile_percent = 1.0 / num_tiles as f32 * 100.0;
        let tile_percent_target = 5.0;
        let tile_divisor = (tile_percent_target / tile_percent).round() as usize;

        if tile_idx % tile_divisor == 0 {
            println!(
                "{}% finished...",
                (tile_idx as f32 * tile_percent).round() as u32
            );
        }

        let mut channels = self.channels.lock().unwrap();

        let Tile {
            channels: tile_channels,
            raster_bounds: tile_bounds,
            ..
        } = tile;

        for (tile_channel, channel) in tile_channels.iter().zip(channels.iter_mut()) {
            // Safe because we guarantee that we won't start modifying this chunk again
            // until the next epoch.
            channel
                .copy_from_tile(tile_channel, self.res, tile_bounds, samples)
                .unwrap();
        }
    }
}

#[inline]
fn sample_uv(
    x: usize,
    y: usize,
    screen_to_ndc_size: Vec2,
    fis: &FilterImportanceSampler,
    rng: &mut SmallRng,
) -> Vec2 {
    let uv_samp = Vec2::new(rng.gen::<f32>(), rng.gen::<f32>());
    let fis_samp = Vec2::new(fis.sample(uv_samp.x), fis.sample(uv_samp.y));

    let screen_coord = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) + fis_samp;

    screen_to_ndc_size * screen_coord
}
