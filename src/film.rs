use generic_array::{ GenericArray, ArrayLength };
use dynamic_arena::{ DynamicArena, NonSend };

use rand::distributions::Uniform;
use rand::prelude::*;

use crate::camera::CameraHandle;
use crate::math::{ Vec2u, Vec2, Vec3, Aabr, Extent2u, Aabru };
use crate::spectrum::{ Xyz, Rgb, IsSpectrum };
use crate::integrator::{ Integrator };
use crate::world::World;

use std::sync::{ atomic::{ AtomicUsize, Ordering }, Mutex };
use std::ops::Range;
use std::cell::UnsafeCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelKind {
    Color,
    Background,
    Normal,
}

impl ChannelKind {
    pub fn channel_size(&self) -> usize {
        match *self {
            ChannelKind::Color => 4,
            ChannelKind::Background => 3,
            ChannelKind::Normal => 3,
        }
    }
}

pub type Channel = (ChannelKind, Vec<f32>);
pub type UnsafeChannel = (ChannelKind, UnsafeCell<Vec<f32>>);

pub type ChannelTileMut<'a> = (ChannelKind, &'a mut [f32]);

unsafe impl<N: ArrayLength<UnsafeChannel>> Send for Tile<N> {}
unsafe impl<N: ArrayLength<UnsafeChannel>> Sync for Tile<N> {}

pub struct Tile<N: ArrayLength<UnsafeChannel>> {
    epoch: AtomicUsize,
    channels: GenericArray<UnsafeChannel, N>,
    pixel_bounds: Aabru,
    uv_bounds: Aabr,
}

pub struct Film<N: ArrayLength<Channel> + ArrayLength<UnsafeChannel>> {
    channels: Mutex<GenericArray<Channel, N>>,
    tiles: Vec<Tile<N>>,
    progressive_epoch: usize,
    this_epoch_tiles_finished: AtomicUsize,
    res: Extent2u,
}

impl<'a, N: ArrayLength<Channel> + ArrayLength<UnsafeChannel>> Film <N> { 
    pub fn new(
        channels: &[ChannelKind],
        res: Extent2u,
    ) -> Self {
        Film {
            channels: Mutex::new(GenericArray::from_exact_iter(channels.into_iter().map(|kind| {
                let size = kind.channel_size();
                (*kind, vec![0f32; size * res.w * res.h])
            })).unwrap()),
            tiles: Vec::new(),
            progressive_epoch: 0,
            this_epoch_tiles_finished: AtomicUsize::new(0),
            res,
        }
    }

    pub fn save_to<P: AsRef<std::path::Path>, IS: Into<String>>(
        &self,
        output_folder: P,
        base_name: IS,
        transparent_background: bool,
    ) {
        let base_name = base_name.into();
        let channels = self.channels.lock().unwrap();
        let (color_idx, bg_idx) = channels
            .iter()
            .enumerate()
            .fold((None, None), |(mut color_idx, mut bg_idx), (i, (kind, _))| {
                match *kind {
                    ChannelKind::Color => color_idx = Some(i),
                    ChannelKind::Background => bg_idx = Some(i),
                    _ => (),
                }
                (color_idx, bg_idx)
            });

        match (color_idx, bg_idx, transparent_background) {
            (Some(color_idx), _, true) => {
                let color_buf = &channels[color_idx].1;
                let mut img = image::RgbaImage::new(self.res.w as u32, self.res.h as u32);
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    let idx = (x as usize + (self.res.h - 1 - y as usize) * self.res.w) * 4;
                    let a = color_buf[idx + 3];
                    let rgb = Rgb::from(Xyz::new(color_buf[idx], color_buf[idx + 1], color_buf[idx + 2]).gamma_corrected(2.2)) * a;
                    *pixel = image::Rgba([
                        (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                        (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                        (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                        (a * 255.0).min(255.0).max(0.0) as u8,
                    ]);
                }
                let filename = output_folder.as_ref().join(format!("{}_color.png", base_name.clone()));
                println!("Saving to {}...", filename.display());
                img.save(filename).unwrap();
            },
            (Some(color_idx), Some(bg_idx), false) => {
                let color_buf = &channels[color_idx].1;
                let bg_buf = &channels[bg_idx].1;
                let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    let i = x as usize + (self.res.h - 1 - y as usize) * self.res.w;
                    let col_idx = i * 4;
                    let bg_idx = i * 3;
                    let col = Xyz::new(color_buf[col_idx], color_buf[col_idx + 1], color_buf[col_idx + 2]);
                    let bg = Xyz::new(bg_buf[bg_idx], bg_buf[bg_idx + 1], bg_buf[bg_idx + 2]);
                    let col = (col + bg).gamma_corrected(2.2);
                    let rgb = Rgb::from(col);
                    *pixel = image::Rgb([
                        (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                        (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                        (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                    ]);
                }
                let filename = output_folder.as_ref().join(format!("{}_color.png", base_name.clone()));
                println!("Saving to {}...", filename.display());
                img.save(filename).unwrap();
            },
            (Some(color_idx), None, false) => {
                let color_buf = &channels[color_idx].1;
                let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                for (x, y, pixel) in img.enumerate_pixels_mut() {
                    let idx = (x as usize + (self.res.h - 1 - y as usize) * self.res.w) * 4;
                    let rgb = Rgb::from(Xyz::new(color_buf[idx], color_buf[idx + 1], color_buf[idx + 2]).gamma_corrected(2.2));
                    *pixel = image::Rgb([
                        (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                        (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                        (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                    ]);
                }
                let filename = output_folder.as_ref().join(format!("{}_color.png", base_name.clone()));
                println!("Saving to {}...", filename.display());
                img.save(filename).unwrap();
            },
            _ => (),
        }

        for (kind, buf) in channels.iter() {
            match *kind {
                ChannelKind::Normal => {
                    let mut img = image::RgbImage::new(self.res.w as u32, self.res.h as u32);
                    for (x, y, pixel) in img.enumerate_pixels_mut() {
                        let idx = (x as usize + (self.res.h - 1 - y as usize) * self.res.w) * 3;
                        let vec = Vec3::new(buf[idx], buf[idx + 1], buf[idx + 2]);
                        let rgb = Rgb::from(vec * 0.5 + Vec3::new(0.5, 0.5, 0.5));
                        *pixel = image::Rgb([
                            (rgb.r * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.g * 255.0).min(255.0).max(0.0) as u8,
                            (rgb.b * 255.0).min(255.0).max(0.0) as u8,
                        ]);
                    }
                    let filename = output_folder.as_ref().join(format!("{}_normal.png", base_name.clone()));
                    println!("Saving to {}...", filename.display());
                    img.save(filename).unwrap();
                },
                _ => (),
            }
        }
    }
}

impl<'a, N: ArrayLength<Channel> + ArrayLength<UnsafeChannel> + ArrayLength<ChannelTileMut<'a>>> Film<N>  {
    pub fn render_frame_into<I: Integrator, S: IsSpectrum>(
        &'a mut self,
        world: &World<S>,
        camera: CameraHandle,
        integrator: I,
        tile_size: Extent2u,
        time_range: Range<f32>,
        samples: usize,
    ) {
        let camera = world.cameras.get(camera);
        self.tiles.clear();

        let rem = Vec2u::new(
            (self.res.w) % tile_size.w,
            (self.res.h) % tile_size.h);
        {
            let channels = self.channels.lock().unwrap();
            for tile_x in 0..((self.res.w + rem.x) / tile_size.w) {
                for tile_y in 0..((self.res.h + rem.y) / tile_size.h) {
                    let start = Vec2u::new(tile_x * tile_size.w, tile_y * tile_size.h);
                    let end = Vec2u::new(
                        (start.x + tile_size.w).min(self.res.w),
                        (start.y + tile_size.h).min(self.res.h));
                    let tile_bounds = Aabru {
                        min: start,
                        max: end,
                    };
                    let actual_tile_size = tile_bounds.size();

                    let uv_start = Vec2::new(start.x as f32 / self.res.w as f32, start.y as f32 / self.res.h as f32);
                    let uv_end = Vec2::new(end.x as f32 / self.res.w as f32, end.y as f32 / self.res.h as f32);
                    let uv_bounds = Aabr {
                        min: uv_start,
                        max: uv_end
                    };

                    let tile_channels = GenericArray::from_exact_iter(channels.iter().map(|(kind, _)| {
                        let size = kind.channel_size();
                        (*kind, UnsafeCell::new(vec![0f32; size * actual_tile_size.w * actual_tile_size.h]))
                    })).unwrap();

                    self.tiles.push(Tile {
                        epoch: AtomicUsize::new(0),
                        channels: tile_channels,
                        pixel_bounds: tile_bounds,
                        uv_bounds,
                    });
                }
            }
        }

        self.integrate_tiles(|mut tile, tile_extent, uv_bounds| {
            let mut rng = rand::thread_rng();
            let uniform = Uniform::new(0.0, 1.0);

            let arena = DynamicArena::<'_, NonSend>::new_bounded();

            let tile_extent_f32 = Vec2::new(tile_extent.w as f32, tile_extent.h as f32);

            for x in 0..tile_extent.w {
                for y in 0..tile_extent.h {
                    let mut col_spect = S::zero();
                    let mut col_a = 0.0;
                    let mut back_spect = S::zero();
                    let mut normals = Vec3::zero();
                    for _ in 0..samples {
                        let r = Vec2::new(uniform.sample(&mut rng), uniform.sample(&mut rng));

                        let tile_xy = Vec2::new(x as f32, y as f32) + r;
                        let uv_offset = tile_xy / tile_extent_f32 * uv_bounds.size();
                        let uv = uv_bounds.min + uv_offset;

                        let time = rng.gen_range(time_range.start, time_range.end);

                        let ray = camera.get_ray(uv, time, &mut rng);
                        integrator.integrate::<S>(
                            world,
                            ray,
                            time,
                            &mut col_spect,
                            &mut col_a,
                            &mut back_spect,
                            &mut normals,
                            &mut rng,
                            &arena);
                    }

                    col_spect = col_spect / samples as f32;
                    col_a = col_a / samples as f32;
                    back_spect = back_spect / samples as f32;
                    normals /= samples as f32;

                    for (kind, buf) in tile.iter_mut() {
                        let size = kind.channel_size();
                        let base_idx = (x + y * tile_extent.w) * size;
                        use ChannelKind::*;
                        match *kind {
                            Color => {
                                let col_xyz: Xyz = col_spect.into();
                                buf[base_idx] = col_xyz.x;
                                buf[base_idx + 1] = col_xyz.y;
                                buf[base_idx + 2] = col_xyz.z;
                                buf[base_idx + 3] = col_a;
                            },
                            Background => {
                                let back_xyz: Xyz = back_spect.into();
                                buf[base_idx] = back_xyz.x;
                                buf[base_idx + 1] = back_xyz.y;
                                buf[base_idx + 2] = back_xyz.z;
                            },
                            Normal => {
                                buf[base_idx] = normals.x;
                                buf[base_idx + 1] = normals.y;
                                buf[base_idx + 2] = normals.z;
                            }
                        }
                    }
                }
            }
        });
    }

    fn integrate_tiles<F>(&'a mut self, integrate_tile: F)
        where F: FnOnce(GenericArray<ChannelTileMut<'a>, N>, Extent2u, Aabr) + Send + Sync + Copy
    {
        let tile_infos = self
            .tiles
            .iter()
            .enumerate()
            .map(|(idx, Tile { channels, pixel_bounds, uv_bounds, .. }) | {
                let tile_channels = GenericArray::from_exact_iter(channels.iter().map(|(kind, buf)| {
                    // Safe because we guarantee that nobody else is accessing this specific
                    // slice at the same time, and we do not modify or read the underlying Vec
                    // until after this ref goes out of scope.
                    let vec = unsafe { &mut *buf.get() };
                    (*kind, vec.as_mut())
                })).unwrap();

                (idx, tile_channels, pixel_bounds.size(), *uv_bounds)
            }).collect::<Vec<_>>();
        
        let num_tiles = self.tiles.len();

        {
            let epoch = self.progressive_epoch;
            let this = &*self;
            rayon::scope_fifo(|scope| {
                for (tile_idx, tile_channels, pixel_size, uv_bounds) in tile_infos.into_iter() {
                    scope.spawn_fifo(move |_| {
                        integrate_tile(tile_channels, pixel_size, uv_bounds);

                        this.tile_finished(tile_idx, epoch)
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

    fn tile_finished(&self, tile_idx: usize, epoch: usize) {
        let tile = &self.tiles[tile_idx];
        let tile_epoch = tile.epoch.fetch_add(1, Ordering::Relaxed);
        if epoch != tile_epoch {
            panic!("Epoch mismatch! Expected: {}, got: {}", tile_epoch, epoch)
        }

        let tile_percent = 1.0 / self.tiles.len() as f32 * 100.0;
        let tile_percent_target = 5.0;
        let tile_divisor = (tile_percent_target / tile_percent).round() as usize;
        
        if tile_idx % tile_divisor == 0 {
            println!(
                "{}% finished...",
                (tile_idx as f32 * tile_percent).round() as u32
            );
        }

        let mut channels = self.channels.lock().unwrap();

        let Tile { channels: tile_channels, pixel_bounds: tile_bounds, .. } = tile;
        let extent = tile.pixel_bounds.size();

        for (tile_channel, channel) in tile_channels.iter().zip(channels.iter_mut()) {
            assert!(tile_channel.0 == channel.0);
            let size = tile_channel.0.channel_size();
            // Safe because we guarantee that we won't start modifying this chunk again
            // until the next epoch.
            let tile_channel = unsafe { &*tile_channel.1.get() };
            for x in 0..extent.w {
                for y in 0..extent.h {
                    let tile_base_idx = (x + y * extent.w) * size;
                    let channel_base_idx = ((tile_bounds.min.x + x) + (tile_bounds.min.y + y) * self.res.w) * size;
                    for offset in 0..size {
                        let tile_idx = tile_base_idx + offset;
                        let channel_idx = channel_base_idx + offset;
                        channel.1[channel_idx] = tile_channel[tile_idx];
                    }
                }
            }
        }
    }
}