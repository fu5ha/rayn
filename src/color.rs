use super::Vec3;
use image;

#[derive(Clone, Copy, Debug)]
pub struct Color(pub Vec3);

impl Color {
    pub fn zero() -> Self {
        Color(Vec3::zero())
    }
}

impl From<Color> for image::Rgb<u8> {
    fn from(col: Color) -> Self {
        image::Rgb {
            data: [(col.0.x * 255.0) as u8, (col.0.y * 255.0) as u8, (col.0.z * 255.0) as u8]
        }
    }
}

impl ::std::ops::Add for Color {
    type Output = Color;

    fn add(self, other: Color) -> Color {
        Color(self.0 + other.0)
    }
}

impl ::std::ops::Div<f32> for Color {
    type Output = Color;

    fn div(self, other: f32) -> Color {
        Color(self.0 / other)
    }
}
