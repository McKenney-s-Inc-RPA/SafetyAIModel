from keeper_secrets_manager_core import SecretsManager
import base64
import os
from base64 import b64decode
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# wrapper for working with a single secret
class SecretsEntry:
    def __init__(self, secret_for_uid):
        self._secret_for_uid = secret_for_uid

    def get_value(self, field_type: str = "password") -> str:
        return self._secret_for_uid.field(field_type, single=True)

    def get_credentials(self):
        return self.get_value("login"), self.get_value("password")


class Keeper:
    client = None
    _is_authorized = False

    @classmethod
    def decrypt_data(cls, data, encryption_key):
        key = b64decode(encryption_key)
        iv = (
            b"\x00" * 16
        )  # 16 bytes for AES block size, set to zeros to match the C# example
        backend = default_backend()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
        decryptor = cipher.decryptor()

        decrypted_data = decryptor.update(data) + decryptor.finalize()

        # Read padding length and remove padding
        padding_length = decrypted_data[-1]
        return decrypted_data[:-padding_length]

    # Using token only to generate the config
    # requires at least one access operation to bind the token

    @classmethod
    def authorize(cls):
        # TODO: use KSM_CONFIG environment variable instead of encrypted ksm_config.json file
        if cls._is_authorized:
            return

        # Retrieve the encryption key from the environment variable
        encryption_key = os.environ.get("KEEPER_ENCRYPTION_KEY")

        # Read the encrypted file path from the environment variable
        encrypted_file_path = os.environ.get("KEEPER_CONFIG_FILE_PATH")

        if not encryption_key or not encrypted_file_path:
            raise Exception(
                "Unable to locate Keeper configuration file! Have you configured this user for Keeper Secrets Manager?"
            )

        # Read the encrypted configuration file
        with open(encrypted_file_path, "rb") as file:
            encrypted_data = file.read()

        # Decrypt the data
        try:
            decrypted_data = cls.decrypt_data(encrypted_data, encryption_key)
        except Exception as e:
            raise Exception(
                f"Unable to decrypt configuration file! Installation may be corrupt. Please re-configure this user for Keeper Secrets Manager. {str(e)}"
            )

        os.environ["KSM_CONFIG"] = base64.b64encode(decrypted_data).decode()

        # When no arguments are provided, the KSM_CONFIG environment variable is used
        cls.client = SecretsManager()
        cls._is_authorized = True

    @classmethod
    def get_secret(cls, record_uid: str) -> SecretsEntry:
        cls.authorize()
        secrets_for_uid = cls.client.get_secrets(record_uid)[0]
        return SecretsEntry(secrets_for_uid)

    @classmethod
    def get_credentials(cls, record_uid: str):
        return cls.get_secret(record_uid).get_credentials()


# wrapper for the secrets manager
# this helps to nudge users toward providing a one-time access code, if needed
class SecretsGetter:
    def __init__(self, config_json_file_name: str):
        # the file name of the json file
        assert isinstance(config_json_file_name, str), "The configuration JSON file name should be a string"
        assert config_json_file_name.endswith(".json"), "The configuration JSON file name should end with .json"

        # # check to see if the file exists
        # # assume that if the file exists,
        # # then the one-time access code has already been provided,
        # # and can initialize using only the file
        # if path_exists(config_json_file_name) and path_isfile(config_json_file_name):
        #     self._secrets_manager = SecretsManager(
        #         config=FileKeyValueStorage(config_json_file_name)
        #     )
        # else:
        #     # assume that since the file does not exist, need to provide a one-time passcode
        #     # get the passcode using the getpass function, to avoid logging it
        #     otp = getpass("Please provide a one-time passcode for the secrets keeper")
        #     self._secrets_manager = SecretsManager(
        #         token=otp,
        #         config=FileKeyValueStorage(config_json_file_name)
        #     )

    def get_secret(self, record_uid: str) -> SecretsEntry:
        Keeper.authorize()
        secrets_for_uid = Keeper.client.get_secrets(record_uid)[0]
        return SecretsEntry(secrets_for_uid)


if __name__ == "__main__":
    u, p2 = Keeper.get_credentials("COCDSFU8R0GD8Rfv_HZ_eQ")
    print("Completed!")
