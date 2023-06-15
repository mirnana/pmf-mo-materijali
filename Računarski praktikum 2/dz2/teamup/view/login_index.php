<?php require_once __SITE_PATH . '/view/_header_login.php'; ?>

<form method="POST" action="<?php echo __SITE_URL; ?>/index.php?rt=login/login">
    Korisničko ime: <input type="text" name="username" />
    <br />
    Lozinka: <input type="password" name="password" />
    <br />
    <button type="submit">Prijava</button>
</form>

<?php require_once __SITE_PATH . '/view/_footer.php'; ?>