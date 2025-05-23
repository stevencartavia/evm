//! EIP-6110 deposit requests parsing

use super::spec::EthExecutorSpec;
use crate::block::BlockValidationError;
use alloc::{string::ToString, vec::Vec};
use alloy_consensus::TxReceipt;
use alloy_primitives::{Address, Bytes, Log};
use alloy_sol_types::{sol, SolEvent};

pub use alloy_eips::eip6110::*;

/// The size of a deposit request in bytes. While the event fields emit
/// bytestrings, those bytestrings are fixed size. The fields are: 48-byte
/// pubkey, 32-byte withdrawal credentials, 8-byte amount, 96-byte signature,
/// and 8-byte index.
const DEPOSIT_BYTES_SIZE: usize = 48 + 32 + 8 + 96 + 8;

sol! {
    #[allow(missing_docs)]
    event DepositEvent(
        bytes pubkey,
        bytes withdrawal_credentials,
        bytes amount,
        bytes signature,
        bytes index
    );
}

/// Accumulate a deposit request from a log. containing a [`DepositEvent`].
pub fn accumulate_deposit_from_log(log: &Log<DepositEvent>, out: &mut Vec<u8>) {
    out.reserve(DEPOSIT_BYTES_SIZE);
    out.extend_from_slice(log.pubkey.as_ref());
    out.extend_from_slice(log.withdrawal_credentials.as_ref());
    out.extend_from_slice(log.amount.as_ref());
    out.extend_from_slice(log.signature.as_ref());
    out.extend_from_slice(log.index.as_ref());
}

/// Accumulate deposits from an iterator of logs.
pub fn accumulate_deposits_from_logs<'a>(
    address: Address,
    logs: impl IntoIterator<Item = &'a Log>,
    out: &mut Vec<u8>,
) -> Result<(), BlockValidationError> {
    logs.into_iter()
        // filter logs by address
        .filter(|log| log.address == address)
        // explicitly filter logs by the DepositEvent's signature hash (first topic)
        .filter(|log| {
            // 0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5
            log.topics().first() == Some(&DepositEvent::SIGNATURE_HASH)
        })
        .try_for_each(|log| {
            // We assume that the log is valid because it was emitted by the
            // deposit contract.
            let decoded_log =
                DepositEvent::decode_log(log).map_err(|err: alloy_sol_types::Error| {
                    BlockValidationError::DepositRequestDecode(err.to_string())
                })?;
            accumulate_deposit_from_log(&decoded_log, out);
            Ok(())
        })
}

/// Accumulate deposits from a receipt. Iterates over the logs in the receipt
/// and accumulates the deposit request bytestrings.
pub fn accumulate_deposits_from_receipt(
    address: Address,
    receipt: impl TxReceipt<Log = Log>,
    out: &mut Vec<u8>,
) -> Result<(), BlockValidationError> {
    accumulate_deposits_from_logs(address, receipt.logs(), out)
}

/// Accumulate deposits from a list of receipts. Iterates over the logs in the
/// receipts and accumulates the deposit request bytestrings.
pub fn accumulate_deposits_from_receipts<'a, I, R>(
    address: Address,
    receipts: I,
    out: &mut Vec<u8>,
) -> Result<(), BlockValidationError>
where
    I: IntoIterator<Item = &'a R>,
    R: TxReceipt<Log = Log> + 'a,
{
    receipts
        .into_iter()
        .try_for_each(|receipt| accumulate_deposits_from_receipt(address, receipt, out))
}

/// Find deposit logs in a list of receipts, and return the concatenated
/// deposit request bytestring.
///
/// The address of the deposit contract is taken from the chain spec, and
/// defaults to [`MAINNET_DEPOSIT_CONTRACT_ADDRESS`] if not specified in
/// the chain spec.
pub fn parse_deposits_from_receipts<'a, I, R>(
    spec: impl EthExecutorSpec,
    receipts: I,
) -> Result<Bytes, BlockValidationError>
where
    I: IntoIterator<Item = &'a R>,
    R: TxReceipt<Log = Log> + 'a,
{
    let mut out = Vec::new();
    accumulate_deposits_from_receipts(
        spec.deposit_contract_address().unwrap_or(MAINNET_DEPOSIT_CONTRACT_ADDRESS),
        receipts,
        &mut out,
    )?;
    Ok(out.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eth::spec::EthSpec;
    use alloc::vec;
    use alloy_consensus::Receipt;
    use alloy_primitives::{b256, bytes};

    #[test]
    fn check_deposit_sig() {
        let expected = b256!("0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5");
        assert_eq!(expected, DepositEvent::SIGNATURE_HASH);
    }

    #[test]
    fn test_parse_deposit_from_log() {
        let receipts = vec![
            // https://etherscan.io/tx/0xa5239d4c542063d29022545835815b78b09f571f2bf1c8427f4765d6f5abbce9
            Receipt {
                status: true.into(),
                cumulative_gas_used: 0,
                logs: serde_json::from_str(
                    r#"[{"address":"0x00000000219ab540356cbb839cbe05303d7705fa","topics":["0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5"],"data":"0x00000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000140000000000000000000000000000000000000000000000000000000000000018000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000030998c8086669bf65e24581cda47d8537966e9f5066fc6ffdcba910a1bfb91eae7a4873fcce166a1c4ea217e6b1afd396200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002001000000000000000000000001c340fb72ed14d4eaa71f7633ee9e33b88d4f3900000000000000000000000000000000000000000000000000000000000000080040597307000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006098ddbffd700c1aac324cfdf0492ff289223661eb26718ce3651ba2469b22f480d56efab432ed91af05a006bde0c1ea68134e0acd8cacca0c13ad1f716db874b44abfcc966368019753174753bca3af2ea84bc569c46f76592a91e97f311eddec0000000000000000000000000000000000000000000000000000000000000008e474160000000000000000000000000000000000000000000000000000000000","blockHash":"0x8d1289c5a7e0965b1d1bb75cdc4c3f73dda82d4ebb94ff5b98d1389cebd53b56","blockNumber":"0x12f0d8d","transactionHash":"0xa5239d4c542063d29022545835815b78b09f571f2bf1c8427f4765d6f5abbce9","transactionIndex":"0xc4","logIndex":"0x18f","removed":false}]"#
                ).unwrap(),
            },
            // https://etherscan.io/tx/0xd9734d4e3953bcaa939fd1c1d80950ee54aeecc02eef6ae8179f47f5b7103338
            Receipt {
                status: true.into(),
                cumulative_gas_used: 0,
                logs: serde_json::from_str(
                    r#"[{"address":"0x00000000219ab540356cbb839cbe05303d7705fa","topics":["0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5"],"data":"0x00000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000140000000000000000000000000000000000000000000000000000000000000018000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000030a1a2ba870a90e889aa594a0cc1c6feffb94c2d8f65646c937f1f456a315ef649533e25a4614d8f4f66ebdb06481b90af0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200100000000000000000000000a0f04a231efbc29e1db7d086300ff550211c2f6000000000000000000000000000000000000000000000000000000000000000800405973070000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000060ad416d590e1a7f52baff770a12835b68904efad22cc9f8ba531e50cbbd26f32b9c7373cf6538a0577f501e4d3e3e63e208767bcccaae94e1e3720bfb734a286f9c017d17af46536545ccb7ca94d71f295e71f6d25bf978c09ada6f8d3f7ba0390000000000000000000000000000000000000000000000000000000000000008e374160000000000000000000000000000000000000000000000000000000000","blockHash":"0x8d1289c5a7e0965b1d1bb75cdc4c3f73dda82d4ebb94ff5b98d1389cebd53b56","blockNumber":"0x12f0d8d","transactionHash":"0xd9734d4e3953bcaa939fd1c1d80950ee54aeecc02eef6ae8179f47f5b7103338","transactionIndex":"0x7c","logIndex":"0xe2","removed":false}]"#,
                ).unwrap(),
            },
        ];

        let request_data = parse_deposits_from_receipts(EthSpec::mainnet(), &receipts).unwrap();
        assert_eq!(
            request_data,
            bytes!(
                "998c8086669bf65e24581cda47d8537966e9f5066fc6ffdcba910a1bfb91eae7a4873fcce166a1c4ea217e6b1afd396201000000000000000000000001c340fb72ed14d4eaa71f7633ee9e33b88d4f39004059730700000098ddbffd700c1aac324cfdf0492ff289223661eb26718ce3651ba2469b22f480d56efab432ed91af05a006bde0c1ea68134e0acd8cacca0c13ad1f716db874b44abfcc966368019753174753bca3af2ea84bc569c46f76592a91e97f311eddece474160000000000a1a2ba870a90e889aa594a0cc1c6feffb94c2d8f65646c937f1f456a315ef649533e25a4614d8f4f66ebdb06481b90af0100000000000000000000000a0f04a231efbc29e1db7d086300ff550211c2f60040597307000000ad416d590e1a7f52baff770a12835b68904efad22cc9f8ba531e50cbbd26f32b9c7373cf6538a0577f501e4d3e3e63e208767bcccaae94e1e3720bfb734a286f9c017d17af46536545ccb7ca94d71f295e71f6d25bf978c09ada6f8d3f7ba039e374160000000000"
            )
        );
    }

    #[test]
    fn test_parse_deposit_from_log_extra() {
        let receipts = vec![
            Receipt {
                status: true.into(),
                cumulative_gas_used: 0,
                // Transfer + Deposit
                logs: serde_json::from_str(
                    r#"[
                      {
    "address": "0x00000000219ab540356cbb839cbe05303d7705fa",
    "topics": [
      "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
      "0x0000000000000000000000008b0c2c4c8eb078bc6c01f48523764c8942c0c6c4",
      "0x0000000000000000000000000000000000000000000000000000000000000000"
    ],
    "data": "0x0000000000000000000000000000000000000000000000000000000000000001",
    "blockNumber": "0x158c89",
    "transactionHash": "0xfcd133f57d5bab8cea211bab0361379456b9115a66bd242d62582aab0bb5fe71",
    "transactionIndex": "0x0",
    "blockHash": "0xe826da725061cdd2ab7638d4c4b5f25a07491f131d05b89628c4613cad07a246",
    "logIndex": "0x0",
    "removed": false
  },
                    {"address":"0x00000000219ab540356cbb839cbe05303d7705fa","topics":["0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5"],"data":"0x00000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000140000000000000000000000000000000000000000000000000000000000000018000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000030998c8086669bf65e24581cda47d8537966e9f5066fc6ffdcba910a1bfb91eae7a4873fcce166a1c4ea217e6b1afd396200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002001000000000000000000000001c340fb72ed14d4eaa71f7633ee9e33b88d4f3900000000000000000000000000000000000000000000000000000000000000080040597307000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000006098ddbffd700c1aac324cfdf0492ff289223661eb26718ce3651ba2469b22f480d56efab432ed91af05a006bde0c1ea68134e0acd8cacca0c13ad1f716db874b44abfcc966368019753174753bca3af2ea84bc569c46f76592a91e97f311eddec0000000000000000000000000000000000000000000000000000000000000008e474160000000000000000000000000000000000000000000000000000000000","blockHash":"0x8d1289c5a7e0965b1d1bb75cdc4c3f73dda82d4ebb94ff5b98d1389cebd53b56","blockNumber":"0x12f0d8d","transactionHash":"0xa5239d4c542063d29022545835815b78b09f571f2bf1c8427f4765d6f5abbce9","transactionIndex":"0xc4","logIndex":"0x18f","removed":false}]"#
                ).unwrap(),
            },
            // https://etherscan.io/tx/0xd9734d4e3953bcaa939fd1c1d80950ee54aeecc02eef6ae8179f47f5b7103338
            Receipt {
                status: true.into(),
                cumulative_gas_used: 0,
                logs: serde_json::from_str(
                    r#"[{"address":"0x00000000219ab540356cbb839cbe05303d7705fa","topics":["0x649bbc62d0e31342afea4e5cd82d4049e7e1ee912fc0889aa790803be39038c5"],"data":"0x00000000000000000000000000000000000000000000000000000000000000a000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000140000000000000000000000000000000000000000000000000000000000000018000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000030a1a2ba870a90e889aa594a0cc1c6feffb94c2d8f65646c937f1f456a315ef649533e25a4614d8f4f66ebdb06481b90af0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000200100000000000000000000000a0f04a231efbc29e1db7d086300ff550211c2f6000000000000000000000000000000000000000000000000000000000000000800405973070000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000060ad416d590e1a7f52baff770a12835b68904efad22cc9f8ba531e50cbbd26f32b9c7373cf6538a0577f501e4d3e3e63e208767bcccaae94e1e3720bfb734a286f9c017d17af46536545ccb7ca94d71f295e71f6d25bf978c09ada6f8d3f7ba0390000000000000000000000000000000000000000000000000000000000000008e374160000000000000000000000000000000000000000000000000000000000","blockHash":"0x8d1289c5a7e0965b1d1bb75cdc4c3f73dda82d4ebb94ff5b98d1389cebd53b56","blockNumber":"0x12f0d8d","transactionHash":"0xd9734d4e3953bcaa939fd1c1d80950ee54aeecc02eef6ae8179f47f5b7103338","transactionIndex":"0x7c","logIndex":"0xe2","removed":false}]"#,
                ).unwrap(),
            },
        ];

        let request_data = parse_deposits_from_receipts(EthSpec::mainnet(), &receipts).unwrap();
        assert_eq!(
            request_data,
            bytes!(
                "998c8086669bf65e24581cda47d8537966e9f5066fc6ffdcba910a1bfb91eae7a4873fcce166a1c4ea217e6b1afd396201000000000000000000000001c340fb72ed14d4eaa71f7633ee9e33b88d4f39004059730700000098ddbffd700c1aac324cfdf0492ff289223661eb26718ce3651ba2469b22f480d56efab432ed91af05a006bde0c1ea68134e0acd8cacca0c13ad1f716db874b44abfcc966368019753174753bca3af2ea84bc569c46f76592a91e97f311eddece474160000000000a1a2ba870a90e889aa594a0cc1c6feffb94c2d8f65646c937f1f456a315ef649533e25a4614d8f4f66ebdb06481b90af0100000000000000000000000a0f04a231efbc29e1db7d086300ff550211c2f60040597307000000ad416d590e1a7f52baff770a12835b68904efad22cc9f8ba531e50cbbd26f32b9c7373cf6538a0577f501e4d3e3e63e208767bcccaae94e1e3720bfb734a286f9c017d17af46536545ccb7ca94d71f295e71f6d25bf978c09ada6f8d3f7ba039e374160000000000"
            )
        );
    }
}
