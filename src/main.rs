use autodiff::*;
use pixel_canvas::{Canvas, Color};

#[derive(Debug, Clone, Copy)]
struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }
}

type SignedDistanceFunction = dyn Fn(&Point) -> FT<f32>;

struct Sdf(Box<SignedDistanceFunction>);

impl Sdf {
    pub fn distance(&self, p: &Point) -> FT<f32> {
        self.0(p)
    }

    pub fn union(self, other: Sdf) -> Sdf {
        Sdf(Box::new(move |p| self.distance(p).min(other.distance(p))))
    }

    pub fn intersection(self, other: Sdf) -> Sdf {
        Sdf(Box::new(move |p| self.distance(p).max(other.distance(p))))
    }

    pub fn invert(self) -> Sdf {
        Sdf(Box::new(move |p| -self.distance(p)))
    }
}

struct Shape;
impl Shape {
    pub fn circle(center: Point, radius: f32) -> Sdf {
        Sdf(Box::new(move |p| {
            (FT::var(center.x - p.x).pow(2.0f32) + FT::var(center.y - p.y).pow(2.0f32)).sqrt()
                - radius
        }))
    }

    pub fn rectangle(min: Point, max: Point) -> Sdf {
        Shape::right(min.x)
            .intersection(Shape::left(max.x))
            .intersection(Shape::upper(min.y))
            .intersection(Shape::lower(max.y))
    }

    pub fn left(x: f32) -> Sdf {
        Sdf(Box::new(move |p| FT::var(p.x - x)))
    }

    pub fn right(x: f32) -> Sdf {
        Sdf(Box::new(move |p| FT::var(x - p.x)))
    }

    pub fn lower(y: f32) -> Sdf {
        Sdf(Box::new(move |p| FT::var(p.y - y)))
    }

    pub fn upper(y: f32) -> Sdf {
        Sdf(Box::new(move |p| FT::var(y - p.y)))
    }
}

fn main() {
    let canvas = Canvas::new(512, 512).title("Tile");
    let h = Shape::rectangle(Point::new(10.0, 10.0), Point::new(20., 70.0))
        .union(
            Shape::rectangle(Point::new(40.0, 10.0), Point::new(50.0, 30.0))
                .union(Shape::circle(Point::new(30.0, 30.0), 20.)),
        )
        .intersection(
            Shape::circle(Point::new(30.0, 30.0), 10.)
                .union(Shape::rectangle(
                    Point::new(20.0, 10.0),
                    Point::new(40.0, 30.0),
                ))
                .invert(),
        );

    let i = Shape::rectangle(Point::new(60.0, 10.0), Point::new(70.0, 40.0))
        .union(Shape::circle(Point::new(65.0, 50.), 5.));

    let sdf = h.union(i);

    let black = Color { r: 0, g: 0, b: 0 };
    let white = Color {
        r: 255,
        g: 255,
        b: 255,
    };

    canvas.render(move |_, image| {
        let width = image.width();
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let distance = sdf
                    .distance(&Point {
                        x: x as f32,
                        y: y as f32,
                    })
                    .x;
                *pixel = if distance < 0.0 { black } else { white }
            }
        }
    });
}
