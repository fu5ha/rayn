use crate::spectrum::{Srgb, WSrgb};
use crate::math::{f32x4, Wec3, Vec3};

#[derive(Clone, Copy)]
pub struct Sky {
    wide_top: WSrgb,
    wide_bottom: WSrgb,
}

impl Sky {
    pub fn new(top: Srgb, bottom: Srgb) -> Self {
        Self {
            wide_top: WSrgb::splat(top),
            wide_bottom: WSrgb::splat(bottom),
        }
    }

    pub fn wide_le(&self, wo: Wec3) -> WSrgb {
        let t = f32x4::from(0.5) * (wo.y + f32x4::ONE);

        self.wide_top * (f32x4::ONE - t) + self.wide_bottom * t
    }

}
