#![no_std]
#![no_main]

/// https://github.com/khoih-prog/RP2040_PWM/blob/main/README.md <- PWM output
/// CONNECTIONS:
///     Battery:
///         12v DC male plug (ciggerette lighter) goes into 12v battery port
///     
///     MDD10A:
///         Power port B- is connected to Negative Anode of cigarette lighter (Black wire)
///         Power port B+ is connected to Positive Cathode of cigarette lighter (Red wire)
///
///         Port M2A is connected to the Cathode of a 12v brush motor
///         Port M2B is connected to the Anode of a 12v brush motor
///         
///         GND is connected to GPIO22 on pico
///         PWM2 is connected to GPIO21 on pico
///         DIR2 is connected to GPIO20 on pico
///         
///         
///     PICO H (rp2040):
///         pico is connected via micro usb to a power 3.3-5v power source when in production
///         pico is connected via micro usb to a power a laptop when in dev, draw from laptop is sufficient
///
///     
///     Drawing:
///         Bat     DVR/Motor    Motor          DVR/CTRL      CTRL          JETSON
///                            _______
///                        ->[-]      \___
///                        | [+]______/
///                [M1B] --]  |
///         [+]    [M1A] ---->|                 [DRV1]   ->   GPIO18
///          |       |                          [PWM1]   ->   GPIO19
///          | ---> [+]                         [DRV2]   ->   GPIO20
///          | ---> [-]<--------------------    [PWM2]   ->   GPIO21
///          |       |                     |     [GRD]   ->   GND
///         [-]    [M2A] ---->|            |                  GPIO0[TX]  ->  RX
///                [M1B] --]  | _____      |             ->   GPIO1[RX]  ->  TX
///                        | [+]      \___ |             ->   VSYS       ->  3.3/5v OUT
///                        ->[-]______/    |                  GND        ->  GND
///                                        ----------------   GND
///
///
/// some helpful screen commands:
/// 1. screen -ls: lists all screens running
/// 2. pkill screen: kills all sessions
/// 3. echo -n "w" > /dev/<target-screen> sets the current active direction

mod motor;

use defmt_rtt as _;
use embedded_hal::digital::StatefulOutputPin;
use panic_probe as _;

use rp2040_hal::pac;
use rp2040_hal::pwm;
use rp2040_hal::uart;
use rp2040_hal::{gpio::{Pins, FunctionUart}, pwm::Slices, Sio, Watchdog};
use rp2040_hal::fugit::RateExtU32;
use rp2040_hal::Clock;


#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

#[cortex_m_rt::entry]
fn main() -> ! {
    // take the peripherals for the device
    let mut pac: pac::Peripherals = pac::Peripherals::take().unwrap();
    // prevents time out
    let mut watchdog: Watchdog = Watchdog::new(pac.WATCHDOG);
    // inialise a clock to keep track of timings
    let clocks: rp2040_hal::clocks::ClocksManager = rp2040_hal::clocks::init_clocks_and_plls(
        12_000_000u32,
        pac.XOSC,
        pac.CLOCKS,
        pac.PLL_SYS,
        pac.PLL_USB,
        &mut pac.RESETS,
        &mut watchdog,
    )
    .unwrap();

    let sio: Sio = Sio::new(pac.SIO);
    // initalise the pins
    let pins: Pins = Pins::new(
        pac.IO_BANK0,
        pac.PADS_BANK0,
        sio.gpio_bank0,
        &mut pac.RESETS,
    );

    let pwm_slices: Slices = Slices::new(pac.PWM, &mut pac.RESETS);

    // sets up two
    let mut motor_one: motor::Motor<pwm::Pwm1> = motor::Motor::from(pins.gpio19, 19, pwm_slices.pwm1);
    let mut motor_two: motor::Motor<pwm::Pwm2> = motor::Motor::from(pins.gpio21, 21, pwm_slices.pwm2);

    // sets gpio25 onboard led to high so that we can see the pico has been flashed.
    let mut led_pin: rp2040_hal::gpio::Pin<
        rp2040_hal::gpio::bank0::Gpio25,
        rp2040_hal::gpio::FunctionSio<rp2040_hal::gpio::SioOutput>,
        rp2040_hal::gpio::PullDown,
    > = pins.gpio25.into_push_pull_output();

    let uart_pins = (
        pins.gpio0.into_function(),
        pins.gpio1.into_function(),
    );

    // setting up uart connections
    // we connect the output
    let uart: uart::UartPeripheral<uart::Enabled, pac::UART0,
        (
            rp2040_hal::gpio::Pin<rp2040_hal::gpio::bank0::Gpio0, FunctionUart, rp2040_hal::gpio::PullDown>,
            rp2040_hal::gpio::Pin<rp2040_hal::gpio::bank0::Gpio1, FunctionUart, rp2040_hal::gpio::PullDown>
        )
    > = uart::UartPeripheral::new(pac.UART0, uart_pins, &mut pac.RESETS)
        .enable(
            uart::UartConfig::new(
                115200.Hz(),
                uart::DataBits::Eight,
                None,
                uart::StopBits::One
            ),
            clocks.peripheral_clock.freq(),
        ).unwrap();

    let (rx, tx) = uart.split();

    let mut rx_buf: [u8; 1] = [0u8; 1];
    let mut count: i32 = 0;
    let mut active_state: &[u8; 1] = b"0";

    loop {
        // 1. We read the rx channel into the rx_buf. This will give us a single byte which will define our status
        // 2. Provided the read is performed correctly and there is data coming from the channel,
        //      we either trigger the motor direction based on the state, OR we continue as normal.
        //      Once we do this we update the active_state to be the direction we are moving, so that on the next
        //      loop iteration we trigger the same direction again.
        match rx.read_raw(&mut rx_buf) {
            Ok(count) if count > 0 => {
                match rx_buf[0] {
                    b'w' if motor_one.state != motor::Direction::FORWARD => {
                        motor_one.forward();
                        motor_two.forward();
                        active_state = b"w";
                    }
                    b's' if motor_one.state != motor::Direction::STOPPED => {
                        motor_one.stop();
                        motor_two.stop();
                        active_state = b"s";
                    }
                    b'x' if motor_one.state != motor::Direction::BACKWARD => {
                        motor_one.backwards();
                        motor_two.backwards();
                        active_state = b"x";
                    }
                    _ => {}
                }
            },
            _ => {}
        }
        // 3. We throttle the uart connection in kinda a cool way by counting up to 5 million
        //      and once we hit the five million we check if the active state is running (not a 0)
        //      and if it isnt then we write what ever value it is.
        //      We do this because rust is so fast it kinda overwhelms the uart connection and
        //      the bytes get written over faster then the pico and pi can keep up. Therefore, triggering
        //      the write command to the pico only activates every 0.5 seconds or so.
        // 4. We flash a led pin to ensure this if block is being triggered.
        count = count.wrapping_add(1);
        if count >= 5_000_000 {
            if active_state != b"0" {
                let _ = tx.write_raw(active_state);
            }
            let _ = led_pin.toggle();
            count = 0;
        }
    }
}
