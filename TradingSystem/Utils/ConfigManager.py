import os

class ConfigManager:
    def __init__(self, data_manager):
        self.data_manager = data_manager

    def get(self, section, key):
        """Get configuration value"""
        credentials = self.data_manager.get_credentials()
        if section == 'exchange':
            if key == 'api_key':
                return credentials['binance_api_key']
            elif key == 'api_secret':
                return credentials['binance_api_secret']
        elif section == 'risk':
            if key == 'max_position_size':
                return float(os.getenv('MAX_POSITION_SIZE', '0.02'))
            elif key == 'max_daily_loss':
                return float(os.getenv('MAX_DAILY_LOSS', '0.05'))
            elif key == 'max_drawdown':
                return float(os.getenv('MAX_DRAWDOWN', '0.10'))
        elif section == 'telegram':
            if key == 'token':
                return credentials['telegram_token']
            elif key == 'chat_id':
                return credentials['telegram_chat_id']
        return None 