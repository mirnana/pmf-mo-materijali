<?php require_once __SITE_PATH . '/view/_header.php'; ?>

<form method="POST" action="<?php echo __SITE_URL . '/index.php?rt=projects/applyForProject'?>">
    Unesi naslov: <input type="text" name="project_title" />
    <br/>
    Unesi opis: <input type="text" name="project_abstract" />
    <br/>
    Unesi ciljani broj članova: <input type="text" name="project_number" />
    <br/>
    <button type="submit">Kreiraj projekt</button>
</form>

<form method="POST" action="<?php echo __SITE_URL; ?>/index.php?rt=login/logout">
    <button type="submit">Odjava</button>
</form>

<?php require_once __SITE_PATH . '/view/_footer.php'; ?>