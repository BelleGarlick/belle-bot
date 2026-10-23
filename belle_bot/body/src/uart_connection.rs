/// UART Connection
///
/// This file is a simple module for establishing a UART (Universal, Asynchronous, Receiver-Transmit)
///
/// The goal here it to establish an object oriented approach to establishing and maintaining
/// connection between mother and child.
///
/// In the example of Belle-bot, the Jetson will send a signal via a pre-defined UART0 pin to
/// the pico micro controller, this signal (b'w', b's' or b'x') will then be ingested by the
/// firmware and directed accordingly. The same will then be sent back so that the Jetson
/// knows the signal has been successful. The two devices will continue to steam this
/// information to one another via the pins until power is lost.
///
use rp2040_hal::clocks::ClocksManager;
use rp2040_hal::fugit::RateExtU32;
use rp2040_hal::pac;
use rp2040_hal::uart;
use rp2040_hal::Clock;

// The UARTConnection object is defined with two read and write pins.
// As the micro controller will be the main device connected via pin
// we will use UART0. This doesn't matter too much but am following convention
//
// In addition to this we parse <P> as a generic, where P has the trait
// ValidUartPinout<UART0> implmeneted, or put simply, the UARTConnection
// will only take pins which are are valid uart pinout or UART0.
pub struct UARTConnection<P>
where
    P: uart::ValidUartPinout<pac::UART0>,
{
    pub rx_pin: uart::Reader<pac::UART0, P>,
    pub tx_pin: uart::Writer<pac::UART0, P>,
}
impl<P> UARTConnection<P>
where
    P: uart::ValidUartPinout<pac::UART0>,
{
    /// Establishes a UART connection between provided uart pins.
    ///
    /// peripherals: UART0 and RESETS parsed from main so we don't have to redefine it.
    /// uart_pins: pins which have been checked are valid UART0.
    /// clocks: parsed from main so we don't have to redefine it.
    /// baudrate: number of times the devices send a signal to each other per second
    pub fn from(
        mut peripherals: (pac::UART0, pac::RESETS),
        uart_pins: P,
        clocks: ClocksManager,
        baudrate: u32,
    ) -> UARTConnection<P> {
        let (rx, tx) = uart::UartPeripheral::new(peripherals.0, uart_pins, &mut peripherals.1)
            .enable(
                uart::UartConfig::new(
                    baudrate.Hz(),         // number of signals sent per second
                    uart::DataBits::Eight, // length of the data frame (1 byte)
                    None,                  // error checking is off
                    uart::StopBits::One,   // signals the end of a data packet
                ),
                clocks.peripheral_clock.freq(), // keeps the rate in sync between devices
            )
            .unwrap()
            .split();

        UARTConnection {
            rx_pin: rx,
            tx_pin: tx,
        }
    }
    /// Reads the incoming message from the mother
    ///
    /// buff: Container the data will be read into
    ///
    /// if the read is successful we return the first
    ///     element in the buffer
    /// else
    ///     we return a null value
    pub fn read(&self, mut buff: [u8; 1]) -> u8 {
        match self.rx_pin.read_raw(&mut buff) {
            Ok(count) if count > 0 => buff[0],
            _ => 0x00,
        }
    }
    /// Writes data to the mother
    ///
    /// count: Number of loops main has been through since transmission
    ///     was last sent.
    /// state: Most recent byte received from mother
    ///
    /// if count is over 5 million and the state is not null,
    ///     we send the transmission for the new state to the mother.
    /// else we parse the current count
    pub fn write(&self, count: i32, state: &[u8]) -> i32 {
        if count >= 5_000_000 {
            if state != b"0" {
                let _ = self.tx_pin.write_raw(state);
            }
            0
        } else {
            count
        }
    }
}
