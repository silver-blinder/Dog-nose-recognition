import antfu from '@antfu/eslint-config'

export default antfu({
    rules: {
        'no-console': 'off',
        'no-unused-vars': 'off',
        'node/prefer-global': 'off',
        'no-alert': 'off',
        'node/prefer-global/buffer': 'off',
        'node/prefer-global/console': 'off',
        'node/prefer-global/process': 'off',
        'node/prefer-global/url': 'off',
        'node/prefer-global/url-search-params': 'off',
    },
})
