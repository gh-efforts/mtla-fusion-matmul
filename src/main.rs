fn find_mask(x: usize, y: usize, window: usize) -> u8 {
    if x > y {
        return 0;
    }

    let x_num = x + 1;
    let y_num = y + 1;
    let mut new_window = window;

    if y_num % 2 != 0 {
        new_window += 1;
    }

    if y_num <= new_window {
        return 1;
    }

    let not_in_window = y_num - new_window;

    if not_in_window < x_num {
        return 1;
    }

    if x_num % 2 != 0 {
        return 0;
    }

    let not_in_vwindow = not_in_window - std::cmp::min(window, not_in_window);

    if not_in_vwindow < x_num {
        return 2;
    }

    if not_in_vwindow < 4 {
        return 2;
    }

    if x_num % 4 == 0 {
        return 4;
    } else {
        return 0;
    }
}

/// output: ((x, y), mask)
fn gen_point_list(mat_rows: usize, window: usize) -> Vec<((usize, usize), u8)> {
    // ((x, y), mask)
    let mut threads = vec![];

    for y in 0..mat_rows {
        for x in 0..mat_rows {
            let mask = find_mask(x, y, window);

            if mask != 0 {
                threads.push(((x, y), mask));
            }
        }
    }
    threads
}

fn mtla_matmul(
    a: &[i32],
    b: &[i32],
    out: &mut [i32],
    tid: usize,
    col_num: usize,
    row_num: usize,
    // ((x, y), mask)
    points: &[((usize, usize), u8)],
) {
    let ((x, y), mask) = points[tid];
    let out_point = &mut out[y * row_num + x];
    let row = &a[y * col_num..y * col_num + col_num];

    for (i, &a_v) in row.iter().enumerate() {
        if mask == 1 {
            let b_v = b[x * col_num + i];
            *out_point += a_v * b_v;
        } else if mask == 2 {
            let b_v = b[(x - 1) * col_num + i] + b[x * col_num + i];
            *out_point += a_v * b_v;
        } else if mask == 4 {
            let b_v = b[(x - 3) * col_num + i] + b[(x - 2) * col_num + i] + b[(x - 1) * col_num + i] + b[x * col_num + i];
            *out_point += a_v * b_v;
        } else {
            panic!("Invalid mask value: {}", mask);
        }
    }
}

fn main() {
    let a = [
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    ];
    let a_ptr = a.as_slice();

    let b = [
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
    ];
    let b_ptr = b.as_slice();

    const T: usize = 24;
    const WINDOW: usize = 6;

    let mut out = [0i32; T * T];
    let out_ptr = out.as_mut_ptr() as usize;
    let out_len = out.len();


    let points = gen_point_list(T, WINDOW);
    let points = points.as_slice();

    rayon::scope(|s| {
        for tid in 0..points.len() {
            s.spawn(move |_| {
                mtla_matmul(
                    a_ptr,
                    b_ptr,
                    unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut i32, out_len) },
                    tid,
                    4, // col_num
                    T, // row_num
                    points,
                )
            });
        }
    });

    println!("out: ");
    for i in 0..T {
        for j in 0..T {
            print!("{}, ", out[i * T + j]);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // output:
    // 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 4, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 4, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    // 0, 0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    // 0, 0, 0, 4, 0, 0, 0, 4, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
    // 0, 0, 0, 4, 0, 0, 0, 4, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    // 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0,
    // 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 0,
    // 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1,

    #[test]
    fn test_find_mask() {
        let t = 24;
        let window = 6;

        for y in 0..t {
            for x in 0..t {
                let mask = find_mask(x, y, 6);
                print!("{}, ", mask)
            }
            println!();
        }
    }

    #[test]
    fn test_gen_point_list() {
        let t = 24;
        let window = 6;

        let points = gen_point_list(t, window);

        for (point, mask) in points {
            println!("({}, {}) -> {}", point.0, point.1, mask);
        }
    }
}
