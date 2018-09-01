use image;
use math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Color(pub Vec3);

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self { Color(Vec3::new(r, g, b)) }
    pub fn zero() -> Self { Color(Vec3::zero()) }
    pub fn gamma_correct(self, gamma: f32) -> Self {
        Color(Vec3::new(self.0.x.powf(1.0/gamma), self.0.y.powf(1.0/gamma), self.0.z.powf(1.0/gamma)))
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

impl ::std::ops::Mul<f32> for Color {
    type Output = Color;

    fn mul(self, other: f32) -> Color {
        Color(self.0 * other)
    }
}

impl ::std::ops::Mul<Color> for Color {
    type Output = Color;

    fn mul(self, other: Color) -> Color {
        Color(self.0 * other.0)
    }
}

