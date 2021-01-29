use comedilib::*;
use log::{debug, error, info, warn};
use ndarray::{s, Array};
use pbr::ProgressBar;
use std::convert::TryInto;
use std::process::exit;

macro_rules! checkerr {
    ($e:expr) => {
        match $e {
            Ok(val) => val,
            Err(msg) => {
                error!("{}", msg);
                exit(-1);
            }
        }
    };
}

fn main() {
    let _e = hdf5::silence_errors();
    env_logger::init();

    let subdevice = 0;
    let bufsz: usize = 2048;
    let amount = 100000;
    let chanlist = vec![
        (0, 0, ARef::Ground),
        (1, 0, ARef::Ground),
        (2, 0, ARef::Ground),
        (3, 0, ARef::Ground),
        (4, 0, ARef::Ground),
        (5, 0, ARef::Ground),
        (6, 0, ARef::Ground),
        (7, 0, ARef::Ground),
        (8, 0, ARef::Ground),
        (9, 0, ARef::Ground),
        (10, 0, ARef::Ground),
        (11, 0, ARef::Ground),
        (12, 0, ARef::Ground),
        (13, 0, ARef::Ground),
        (14, 0, ARef::Ground),
        (15, 0, ARef::Ground),
    ];

    let mut comedi = checkerr!(Comedi::open(0));
    let subdev_flags = checkerr!(comedi.get_subdevice_flags(subdevice));
    let sample_bytes = if subdev_flags.is_set(SDF::LSAMPL) {
        std::mem::size_of::<LSampl>()
    } else {
        std::mem::size_of::<Sampl>()
    };
    info!("Board: {}", checkerr!(comedi.get_board_name()));
    info!("Driver: {}", checkerr!(comedi.get_driver_name()));
    info!("Device uses {}-byte samples", sample_bytes);
    match comedi.set_buffer_size(
        subdevice,
        (bufsz * chanlist.len() * sample_bytes).try_into().unwrap()
    ) {
        Err(msg) => warn!("Couldn't set buffer size: {}", msg),
        Ok(size) => info!("Kernel ring buffer set to: {} bytes", size),
    };
    let mut cmd =
        checkerr!(comedi.get_cmd_generic_timed(subdevice, chanlist.len().try_into().unwrap(), 300));
    cmd.set_chanlist(&chanlist);
    cmd.set_stop(StopTrigger::Count, amount);

    for i in 0..3 {
        match checkerr!(comedi.command_test(&mut cmd)) {
            CommandTestResult::Ok => {
                info!("Command test succeeded!");
                print_cmd(&cmd);
                break;
            }
            oth => {
                warn!("Test failed with: {:?}", oth);
                print_cmd(&cmd);
                if i == 2 {
                    error!("Test failed too many times. Exiting.");
                    exit(-1);
                }
            }
        };
    }

    checkerr!(comedi.set_read_subdevice(subdevice));
    checkerr!(comedi.command(&cmd));

    let mut total = 0;
    let file = checkerr!(hdf5::File::create("out.h5"));

    let mut leftovers = 0;
    let mut pb = ProgressBar::new(amount.into());
    let mut spins = 0;

    if subdev_flags.is_set(SDF::LSAMPL) {
        let mut buf = Array::zeros((bufsz, chanlist.len()));
        let dset = checkerr!(file
            .new_dataset::<LSampl>()
            .create("dset", (amount as usize, chanlist.len())));
        loop {
            let read_s = leftovers + if checkerr!(comedi.get_buffer_contents(subdevice)) > 0 {
                checkerr!(comedi.read_sampl::<LSampl>(&mut buf.as_slice_mut().unwrap()[leftovers..]))
            } else {
                0
            };
            if read_s == 0 {
                if total as u32 != amount {
                    error!("Device stopped before providing all data");
                    exit(-1);
                }
                break;
            }
            let read = read_s / chanlist.len();
            leftovers = read_s % chanlist.len();
            checkerr!(dset.write_slice(buf.slice(s![..read, ..]), s![total..total + read, ..]));
            for i in 0..leftovers {
                buf[[0, i]] = buf[[read, i]];
            }
            total += read;
            pb.add(read.try_into().unwrap());
        }
    } else {
        let mut buf = Array::zeros((bufsz, chanlist.len()));
        let dset = checkerr!(file
            .new_dataset::<Sampl>()
            .create("dset", (amount as usize, chanlist.len())));
        loop {
            let read_s = leftovers + if checkerr!(comedi.get_buffer_contents(subdevice)) > 0 {
                checkerr!(comedi.read_sampl::<Sampl>(&mut buf.as_slice_mut().unwrap()[leftovers..]))
            } else {
                0
            };
            if read_s == 0 {
                if total as u32 != amount {
                    if spins > 5 {
                        error!("Device stopped before providing all data");
                        exit(-1);
                    }
                    spins += 1;
                } else {
                    break;
                }
            } else {
                spins = 0;
            }
            let read = read_s / chanlist.len();
            leftovers = read_s % chanlist.len();
            checkerr!(dset.write_slice(buf.slice(s![..read, ..]), s![total..total + read, ..]));
            for i in 0..leftovers {
                buf[[0, i]] = buf[[read, i]];
            }
            total += read;
            pb.add(read.try_into().unwrap());
        }
    }
    pb.finish();
    info!("Done");
}

fn print_cmd(cmd: &Cmd) {
    debug!(
        "Command:\nsubdev:\t{}\nstart:\t{:?}\t{}\nscan_begin:\t{:?}\t{}\n\
        convert:\t{:?}\t{}\nscan_end:\t{:?}\t{}\nstop:\t{:?}\t\
        {}\nchanlist:\t{:?}",
        cmd.subdev(),
        cmd.start_src(),
        cmd.start_arg(),
        cmd.scan_begin_src(),
        cmd.scan_begin_arg(),
        cmd.convert_src(),
        cmd.convert_arg(),
        cmd.scan_end_src(),
        cmd.scan_end_arg(),
        cmd.stop_src(),
        cmd.stop_arg(),
        cmd.chanlist().unwrap()
    );
}
