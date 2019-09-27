use generic_array::{ GenericArray, ArrayLength };
use dynamic_arena::{ DynamicArena, NonSend };

use rand::distributions::Uniform;
use rand::prelude::*;

use crate::camera::CameraHandle;
use crate::math::{ Vec2u, Vec2, Vec3, Aabr, Extent2u, Aabru };
use crate::spectrum::{ Xyz, Rgb, IsSpectrum };
use crate::integrator::{ Integrator };
use crate::world::World;

use std::sync::Mutex;
use std::ops::Range;

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

pub type ChannelChunkMut<'a> = (ChannelKind, &'a mut [f32]);

pub struct Film<N: ArrayLength<Channel>> {
    channels: Mutex<GenericArray<Channel, N>>,
    chunks: (usize, Vec<(GenericArray<Channel, N>, Aabru, Aabr)>),
    res: Extent2u,
}

struct Finished {
    pub epoch: usize,
    pub chunk_idx: usize,
}

impl<'a, N: ArrayLength<Channel> + ArrayLength<ChannelChunkMut<'a>>> Film<N>  {
    pub fn new(
        channels: &[ChannelKind],
        res: Extent2u,
    ) -> Self {
        Film {
            channels: Mutex::new(GenericArray::from_exact_iter(channels.into_iter().map(|kind| {
                let size = kind.channel_size();
                (*kind, vec![0f32; size * res.w * res.h])
            })).unwrap()),
            chunks: (0, Vec::new()),
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
                    let rgb = Rgb::from(Xyz::new(color_buf[idx], color_buf[idx + 1], color_buf[idx + 2]).gamma_corrected(2.2));
                    let a = color_buf[idx + 3];
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

    pub fn render_frame_into<I: Integrator, S: IsSpectrum>(
        &mut self,
        world: &World<S>,
        camera: CameraHandle,
        integrator: I,
        chunk_size: Extent2u,
        time_range: Range<f32>,
        samples: usize,
    ) {
        let camera = world.cameras.get(camera);

        self.iter_chunks(chunk_size, |mut chunk, chunk_extent, uv_bounds| {
            let mut rng = rand::thread_rng();
            let uniform = Uniform::new(0.0, 1.0);

            let arena = DynamicArena::<'_, NonSend>::new_bounded();

            let chunk_extent_f32 = Vec2::new(chunk_extent.w as f32, chunk_extent.h as f32);

            for x in 0..chunk_extent.w {
                for y in 0..chunk_extent.h {
                    let mut col_spect = S::zero();
                    let mut col_a = 0.0;
                    let mut back_spect = S::zero();
                    let mut normals = Vec3::zero();
                    for _ in 0..samples {
                        let r = Vec2::new(uniform.sample(&mut rng), uniform.sample(&mut rng));

                        let chunk_xy = Vec2::new(x as f32, y as f32) + r;
                        let uv_offset = chunk_xy / chunk_extent_f32 * uv_bounds.size();
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

                    for (kind, buf) in chunk.iter_mut() {
                        let size = kind.channel_size();
                        let base_idx = (x + y * chunk_extent.w) * size;
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

    fn iter_chunks<F>(&mut self, canonical_chunk_size: Extent2u, integrate_chunk: F)
        where F: FnOnce(GenericArray<ChannelChunkMut<'a>, N>, Extent2u, Aabr) + Send + Sync + Copy
    {
        self.chunks.0 += 1;
        self.chunks.1.clear();

        let rem = Vec2u::new(
            (self.res.w) % canonical_chunk_size.w,
            (self.res.h) % canonical_chunk_size.h);
        {
            let channels = self.channels.lock().unwrap();
            for cx in 0..((self.res.w + rem.x) / canonical_chunk_size.w) {
                for cy in 0..((self.res.h + rem.y) / canonical_chunk_size.h) {
                    let start = Vec2u::new(cx * canonical_chunk_size.w, cy * canonical_chunk_size.h);
                    let end = Vec2u::new(
                        (start.x + canonical_chunk_size.w).min(self.res.w),
                        (start.y + canonical_chunk_size.h).min(self.res.h));
                    let chunk_bounds = Aabru {
                        min: start,
                        max: end,
                    };
                    let chunk_size = chunk_bounds.size();
                    let uv_start = Vec2::new(start.x as f32 / self.res.w as f32, start.y as f32 / self.res.h as f32);
                    let uv_end = Vec2::new(end.x as f32 / self.res.w as f32, end.y as f32 / self.res.h as f32);
                    let uv_bounds = Aabr {
                        min: uv_start,
                        max: uv_end
                    };

                    let channel_chunks = GenericArray::from_exact_iter(channels.iter().map(|(kind, _)| {
                        let size = kind.channel_size();
                        (*kind, vec![0f32; size * chunk_size.w * chunk_size.h])
                    })).unwrap();

                    self.chunks.1.push((
                        channel_chunks, chunk_bounds, uv_bounds
                    ));
                }
            }
        }

        let chunk_buffers = self
            .chunks
            .1
            .iter_mut()
            .map(|(channel_chunks, _, _)| {
                GenericArray::from_exact_iter(channel_chunks.iter_mut().map(|(kind, buf)| {
                    let buffer_ptr = buf.as_mut_ptr();
                    let buffer_len = buf.len();
                    // Safe because we guarantee that nobody else is accessing this specific
                    // slice at the same time, and we do not modify or read the underlying Vec
                    // until after this ref goes out of scope.
                    (*kind, unsafe { std::slice::from_raw_parts_mut(buffer_ptr, buffer_len) })
                })).unwrap()
            }).collect::<Vec<_>>();

        let this = &*self;
        rayon::scope_fifo(|scope| {
            let epoch = self.chunks.0;
            for (chunk_idx, chunk_buffer) in chunk_buffers.into_iter().enumerate() {
                scope.spawn_fifo(move |_| {
                    let (_, bounds, uv_bounds) = this.chunks.1[chunk_idx];
                    integrate_chunk(chunk_buffer, bounds.size(), uv_bounds);
                    
                    this.chunk_finished(Finished { epoch, chunk_idx });
                })
            }
        })
    }

    fn chunk_finished(&self, finished: Finished) {
        if finished.epoch != self.chunks.0 {
            panic!("Epoch mismatch! Expected: {}, got: {}", self.chunks.0, finished.epoch)
        }

        let chunk_percent = 1.0 / self.chunks.1.len() as f32 * 100.0;
        let chunk_percent_target = 5.0;
        let chunk_divisor = (chunk_percent_target / chunk_percent).round() as usize;
        
        if finished.chunk_idx % chunk_divisor == 0 {
            println!(
                "{}% finished...",
                (finished.chunk_idx as f32 * chunk_percent).round() as u32
            );
        }

        let mut channels = self.channels.lock().unwrap();

        let (channel_chunks, bounds, _) = &self.chunks.1[finished.chunk_idx];

        let extent = bounds.size();

        for (chunk, channel) in channel_chunks.iter().zip(channels.iter_mut()) {
            assert!(chunk.0 == channel.0);
            let size = chunk.0.channel_size();
            for x in 0..extent.w {
                for y in 0..extent.h {
                    let chunk_base_idx = (x + y * extent.w) * size;
                    let channel_base_idx = ((bounds.min.x + x) + (bounds.min.y + y) * self.res.w) * size;
                    for offset in 0..size {
                        let chunk_idx = chunk_base_idx + offset;
                        let channel_idx = channel_base_idx + offset;
                        channel.1[channel_idx] = chunk.1[chunk_idx];
                    }
                }
            }
        }
    }
}