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
mod setup;
mod types;
mod uart_connection;

use embedded_hal::digital::OutputPin as _;
use setup as mysetup;
use uart_connection::UARTConnection;

use defmt_rtt as _;
use panic_probe as _;

use rp2040_hal::gpio;
use rp2040_hal::gpio::bank0;
use rp2040_hal::pac;
use rp2040_hal::pwm;
use rp2040_hal::uart;

#[link_section = ".boot2"]
#[used]
pub static BOOT2: [u8; 256] = rp2040_boot2::BOOT_LOADER_W25Q080;

#[cortex_m_rt::entry]
fn main() -> ! {
    // Setup pins and clocks
    let (pins, clocks) = mysetup::setup_pins();

    let mut pac: pac::Peripherals = pac::Peripherals::take().unwrap();
    let pwm_slices: pwm::Slices = pwm::Slices::new(pac.PWM, &mut pac.RESETS);
    let uart_peripherals: (pac::UART0, pac::RESETS) = (pac.UART0, pac.RESETS);

    // PWM Motor set up:
    let mut motor_one: motor::Motor<
        pwm::Pwm1,
        gpio::Pin<bank0::Gpio18, gpio::FunctionNull, gpio::PullDown>,
        gpio::Pin<bank0::Gpio19, gpio::FunctionNull, gpio::PullDown>,
    > = motor::Motor::from(pins.gpio18, pins.gpio19, pwm_slices.pwm1);
    let mut motor_two: motor::Motor<
        pwm::Pwm2,
        gpio::Pin<bank0::Gpio20, gpio::FunctionNull, gpio::PullDown>,
        gpio::Pin<bank0::Gpio21, gpio::FunctionNull, gpio::PullDown>,
    > = motor::Motor::from(pins.gpio20, pins.gpio21, pwm_slices.pwm2);

    // UART serial mother child connection:
    let uart_pins: types::UartPins = uart::Pins::default()
        .tx(pins.gpio0.into_function::<gpio::FunctionUart>())
        .rx(pins.gpio1.into_function::<gpio::FunctionUart>());
    let uart_connection: UARTConnection<uart::Pins<_, _, _, _>> =
        UARTConnection::from(uart_peripherals, uart_pins, clocks, 115200);

    // sets gpio25 onboard led to high so that we can see the pico has been flashed.
    pins.gpio25.into_push_pull_output().set_high();

    // initialise read buffer
    let rx_buf: [u8; 1] = [0u8; 1];
    let mut count: i32 = 0;

    loop {
        // 1. We read the rx channel into the rx_buf. This will give us a single byte which will define our status.
        let command: u8 = uart_connection.read(rx_buf);
        // 2. We exit early if we read nothing from the buffer
        if command == 0x00 {
            continue;
        }
        // 3. We then resolve the u8 into the proper direction and speed.
        //    if the u8 is between 97 and 107 we push forwards
        //    if the u8 is between 107 and 117 we go back
        //    for anything else we stop
        let speed_and_direction: motor::Speed = motor::Speed::from_buff(command);
        // 4. we match the direction and set the speed coming in.
        //    we also set the active status to the string slice of what ever the command is
        //    We do this so we can echo back that the message has been received correctly and applied.
        let active_state: &[u8; 1] = match speed_and_direction.get_direction() {
            motor::Direction::Forward => {
                motor_one.forward(speed_and_direction.get_rate());
                motor_two.forward(speed_and_direction.get_rate());
                &[command]
            }
            motor::Direction::Stopped => {
                motor_one.stop();
                motor_two.stop();
                &[command]
            }
            motor::Direction::Backward => {
                motor_one.backwards(speed_and_direction.get_rate());
                motor_two.backwards(speed_and_direction.get_rate());
                &[command]
            }
            motor::Direction::Left => {
                motor_one.stop();
                motor_two.forward(speed_and_direction.get_rate());
                &[command]
            }
            motor::Direction::Right => {
                motor_one.forward(speed_and_direction.get_rate());
                motor_two.stop();
                &[command]
            }
        };
        // 5. We throttle the uart connection in kinda a cool way by counting up to 5 million
        //      and once we hit the five million we check if the active state is running (not a 0)
        //      and if it isnt then we write what ever value it is.
        //      We do this because rust is so fast it kinda overwhelms the uart connection and
        //      the bytes get written over faster then the pico and pi can keep up. Therefore, triggering
        //      the write command to the pico only activates every 0.5 seconds or so.
        count = count.wrapping_add(1);
        count = uart_connection.write(count, active_state);
    }
}
