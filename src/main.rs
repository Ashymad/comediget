use log::{debug, info, warn, error};
use comedilib::*;
use std::convert::TryInto;
use ndarray::{s, Array};
use std::os::raw::c_double;

fn main() {
    let _e = hdf5::silence_errors();
    env_logger::init();

    let subdevice = 0;
    let bufsz =  1000;
    let amount = 10000;
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

    let mut comedi = Comedi::open(0).unwrap();
    let mut cmd = comedi
        .get_cmd_generic_timed(subdevice, chanlist.len().try_into().unwrap(), 10)
        .unwrap();
    cmd.set_chanlist(&chanlist);
    cmd.set_stop(StopTrigger::Count, amount);

    for i in 0..3 {
        match comedi.command_test(&mut cmd).unwrap() {
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
                    return;
                }
            }
        };
    }

    comedi.set_read_subdevice(subdevice).unwrap();

    let subdev_flags = comedi.get_subdevice_flags(subdevice).unwrap();

    let mut total = 0;
    let file = hdf5::File::create("out.h5").unwrap();
    set_global_oor_behavior(OORBehavior::NaN);
    let range = comedi.get_range(0, 0, 0).unwrap();
    let maxdata = comedi.get_maxdata(0, 0).unwrap();

    comedi.command(&cmd).unwrap();

    let mut lbuf = Array::zeros((bufsz, chanlist.len()));
    let mut buf = Array::zeros((bufsz, chanlist.len()));
    let mut physbuf = Array::zeros((bufsz, chanlist.len()));

    let dset = file.new_dataset::<c_double>().create("dset", (amount as usize, chanlist.len())).unwrap();
    let mut leftovers = 0;
    loop {
        let read_s = if subdev_flags.is_set(SDF::LSAMPL) {
            let read_s = comedi.read_sampl::<LSampl>(lbuf.as_slice_mut().unwrap()).unwrap();
            lbuf.iter().zip(physbuf.iter_mut().skip(leftovers)).for_each(|(lbuf_el, physbuf_el)| *physbuf_el = to_phys(*lbuf_el, &range, maxdata).unwrap());
            read_s + leftovers
        } else {
            let read_s = comedi.read_sampl::<Sampl>(buf.as_slice_mut().unwrap()).unwrap();
            buf.iter().zip(physbuf.iter_mut().skip(leftovers)).for_each(|(buf_el, physbuf_el)| *physbuf_el = to_phys(*buf_el as LSampl, &range, maxdata).unwrap());
            read_s + leftovers
        };
        if read_s == 0 { break; }
        let read = read_s / chanlist.len();
        leftovers = read_s % chanlist.len();
        info!("Read {}/{} samples", total+read, amount);
        dset.write_slice(physbuf.slice(s![..read,..]), s![total..total+read,..]).unwrap();
        for i in 0..leftovers {
            physbuf[[0,i]] = physbuf[[read,i]];
        }
        total += read;
    }
}

fn print_cmd(cmd: &Cmd) {
    debug!(
        "Cmd {{subdev: {}, start_src: {:?}, start_arg: {}, scan_begin_src: {:?}, scan_begin_arg: {}, \
        convert_src: {:?}, convert_arg: {}, scan_end_src: {:?}, scan_end_arg: {}, stop_src: {:?}, \
        stop_arg: {}, chanlist: {:?}}}",
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
        cmd.chanlist()
    );
}
