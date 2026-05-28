import { test, expect } from '@playwright/test';

// Функция для сбора данных о радиокнопках и чекбоксах
async function collectFormElements(page, radioGroupName) {
  // Собираем радиокнопки из указанной группы
  //const radioButtons = page.locator(`input[type="radio"][name="${radioGroupName}"]`);
  const radioButtons = page.locator('#menu-dataset-var-view-main').locator(`input[type="radio"][name="${radioGroupName}"]`);
  const radioCount = await radioButtons.count();
  const radios = [];

  for (let i = 0; i < radioCount; i++) {
    const radio = radioButtons.nth(i);
    const value = await radio.getAttribute('value') || `radio-${i}`;
    radios.push(value);
  }

  // Собираем все чекбоксы 
  //const checkboxes = page.locator('input[type="checkbox"]');
  const checkboxes = page.locator('#menu-dataset-var-view-main').locator('input[type="checkbox"]');
  const checkboxCount = await checkboxes.count();
  const checkboxIds = [];

  for (let i = 0; i < checkboxCount; i++) {
    const checkbox = checkboxes.nth(i);
    const id = await checkbox.getAttribute('id') || `checkbox-${i}`;
    checkboxIds.push(id);
  }

  return { radios, checkboxIds };
}

// Генерация всех комбинаций
function generateCombinations(radios, checkboxIds) {
  const combinations = [];

  // Для каждой радиокнопки генерируем комбинации с чекбоксами
  for (const radioValue of radios) {
    // 2^n комбинаций для чекбоксов (каждый может быть 0 или 1)
    const totalCheckboxCombinations = Math.pow(2, checkboxIds.length);

    for (let i = 0; i < totalCheckboxCombinations; i++) {
      const checkboxCombination = {};

      // Генерируем состояние каждого чекбокса
      for (let j = 0; j < checkboxIds.length; j++) {
        checkboxCombination[checkboxIds[j]] = (i >> j) & 1; // 0 или 1
      }

      combinations.push({
        selectedRadio: radioValue,
        checkboxStates: checkboxCombination
      });
    }
  }

  return combinations;
}

async function goAllCombinations(page, menuItem, menuItemHover) {

  // Собираем элементы формы
  const { radios, checkboxIds } = await collectFormElements(page, 'dataset-variable-labelings');

  console.log(`Радиокнопки: ${radios}`);
  console.log(`Чекбоксы: ${checkboxIds}`);

  // Генерируем все комбинации
  const allCombinations = generateCombinations(radios, checkboxIds);
  console.log(`Всего комбинаций: ${allCombinations.length}`);

  // Применяем каждую комбинацию
  for (let i = 0; i < allCombinations.length; i++) {
    console.log(`--- Комбинация ${i + 1}/${allCombinations.length} ---`);
    console.log('Радиокнопка:', allCombinations[i].selectedRadio);
    console.log('Чекбоксы:', allCombinations[i].checkboxStates);

    try {

      await applyCombination(page, allCombinations[i]);

      await page.getByRole('button', { name: 'Accept' }).click();
      await page.waitForResponse('**/dataset', { timeout: 1000 });

      if (i < allCombinations.length - 1) {
        await page.locator('#menu-dataset-var-view-accept').click();
      } else {
        await page.locator('#menu-dataset-view-accept').click();
        await page.locator('#menu-dataset-view-main').getByText('example').hover();
      }
      
    } catch (error) {
      console.error(`Ошибка в комбинации: ${i + 1}/${allCombinations.length} `);
      console.error('Радиокнопка:', allCombinations[i].selectedRadio);
      console.error('Чекбоксы:', allCombinations[i].checkboxStates);
 
      if (i < allCombinations.length - 1) {

        await page.reload();
        await page.locator('#menu-dataset-view-main').getByText('example').hover();
      
        if (menuItemHover) {
          await menuItemHover.hover();
        }
        
        await menuItem.click();

        await page.getByRole('button', { name: 'Accept' }).click();
        await page.waitForResponse('**/dataset');
      } else {
        throw error;
      }

    }

  }

}

// Применение конкретной комбинации
async function applyCombination(page, combination) {
  // Выбираем радиокнопку
  await page.locator(
    `input[type="radio"][value="${combination.selectedRadio}"]`
  ).check();

  // Устанавливаем состояния чекбоксов
  for (const [checkboxId, isChecked] of Object.entries(combination.checkboxStates)) {
    const checkbox = page.locator(`input[type="checkbox"][id="${checkboxId}"]`);

    if (isChecked) {
      await checkbox.check();
    } else {
      await checkbox.uncheck();
    }
  }
}

// Основной скрипт
test('test datasets 2', async ({ page }) => {
    await page.goto('http://10.10.53.186:8090/');

    await page.locator('#menu-dataset-view-main').getByText('example').hover();

    // Находим все видимые подменю
    const visibleSubmenus = page.locator('.submenu[style*="display: block"]');
    //const submenuCount = await visibleSubmenus.count();

    const submenu = visibleSubmenus.nth(0);
    const menuItems = submenu.locator('.dropdownmenuitem');
    const itemCount = await menuItems.count();

    console.log(`В Подменю 1 уровня: найдено ${itemCount} пунктов`);

    for (let itemIdx = 0; itemIdx < itemCount; itemIdx++) {

      try {
        
        const menuItem = menuItems.nth(itemIdx);

        // Получаем текст пункта
        const itemText = await menuItem.textContent();

        console.log(`  Навели на пункт: ${itemText?.trim()}`);
        await menuItem.hover();

        const visibleSubmenus = page.locator('.submenu[style*="display: block"]');
        const submenuCount = await visibleSubmenus.count();

        if (submenuCount === 2) {

          const submenu = visibleSubmenus.nth(1);
          const menuItems = submenu.locator('.dropdownmenuitem');
          const itemCount = await menuItems.count();

          console.log(`Подменю: найдено ${itemCount} пунктов`);

          for (let itemIdx = 0; itemIdx < itemCount; itemIdx++) {
              const menuItem2 = menuItems.nth(itemIdx);

              // Получаем текст пункта
              const itemText = await menuItem2.textContent();

              console.log(`  Кликаем на пункт: ${itemText?.trim()}`);

              await menuItem2.click();

              await page.getByRole('button', { name: 'Accept' }).click();
              await page.waitForResponse('**/dataset', { timeout: 1000 });

              await goAllCombinations(page, menuItem2, menuItem);
          }

        } else {

          console.log(`  Кликаем на пункт: ${itemText?.trim()}`);

          await menuItem.click();

          await page.getByRole('button', { name: 'Accept' }).click();
          await page.waitForResponse('**/dataset', { timeout: 1000 });
        
          await goAllCombinations(page, menuItem);

        }

      } catch(error) {
        console.error(`Ошибка в пункте меню : ${itemIdx + 1}`);

        await page.reload();
        await page.locator('#menu-dataset-view-main').getByText('example').hover();

      }
             
    }
 
});
